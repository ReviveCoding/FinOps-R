from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactVerificationError,
    artifact_manifest,
    atomic_json,
    publish_verified_tree,
    sha256_file,
    verify_artifact_manifest,
)
from .json_safety import require_json_object, strict_json_load
from .orchestration_stage3 import (
    OrchestrationState,
    resume_human_review,
    run_direct_orchestrator,
    trace_contract_is_complete,
)
from .planner import ActionPacket, PlannerRequest, hold_packet
from .policy_cards import compile_policy_cards, load_compiled_policy_cards
from .retrieval import RetrievalHit
from .verification_policy import CLAIM_BOUNDARY, PolicyContext


STAGE3_BENCHMARK_SCHEMA_VERSION = "1.0"


def _load_scenarios(path: Path) -> dict[str, Any]:
    raw = require_json_object(strict_json_load(path), source=f"Stage 3 scenarios {path}")
    if set(raw) != {"schema_version", "development_only", "scenarios"}:
        raise ValueError("Stage 3 scenario bundle schema is incompatible")
    if raw["schema_version"] != STAGE3_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("Stage 3 scenario schema version is incompatible")
    if raw["development_only"] is not True:
        raise ValueError("Stage 3 scenarios must be labeled development-only")
    scenarios = raw["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Stage 3 scenarios must be one non-empty list")
    ids = [item.get("scenario_id") for item in scenarios if isinstance(item, dict)]
    if len(ids) != len(scenarios) or len(ids) != len(set(ids)):
        raise ValueError("Stage 3 scenario IDs must be unique")
    return raw


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    matches = sum(bool(item["expectation_match"]) for item in results)
    trace_complete = sum(bool(item["state"]["trace_complete"]) for item in results)
    external_execution = sum(
        bool(item["state"]["receipt"] and item["state"]["receipt"]["external_execution"])
        for item in results
    )
    rejected_not_held = sum(
        item["state"]["verification"]["status"] == "REJECTED" and item["state"]["route"] != "HOLD"
        for item in results
    )
    receipts = [item["state"]["receipt"] for item in results if item["state"]["receipt"]]
    invalid_target = sum(receipt["target"] != "TRADEOPS_SIM" for receipt in receipts)
    return {
        "schema_version": STAGE3_BENCHMARK_SCHEMA_VERSION,
        "development_only": True,
        "scenario_count": count,
        "expectation_match_rate": matches / count,
        "trace_complete_rate": trace_complete / count,
        "external_execution_count": external_execution,
        "invalid_simulation_target_count": invalid_target,
        "verifier_rejection_not_held_count": rejected_not_held,
        "simulation_receipt_count": len(receipts),
        "final_route_counts": {
            route: sum(item["state"]["route"] == route for item in results)
            for route in ("SIMULATE", "REVIEW", "HOLD")
        },
        "status": (
            "PASS"
            if matches == count
            and trace_complete == count
            and external_execution == 0
            and invalid_target == 0
            and rejected_not_held == 0
            else "FAIL"
        ),
    }


def run_stage3_benchmark(
    *,
    source_cards_path: Path,
    scenarios_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    scenarios_source = _load_scenarios(scenarios_path)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.staging.",
        )
    )
    try:
        compile_policy_cards(source_cards_path, staging)
        cards = load_compiled_policy_cards(staging / "policy_cards.json")
        cards_by_id = {card.card_id: card for card in cards}
        results: list[dict[str, Any]] = []
        for raw in scenarios_source["scenarios"]:
            scenario_id = raw["scenario_id"]
            request = PlannerRequest(
                case_id=f"STAGE3-{scenario_id}",
                decision_time=datetime(2026, 1, 15, 20, tzinfo=UTC),
                predicted_cause=raw["predicted_cause"],
                candidate_action=raw["candidate_action"],
                permitted_actions=tuple(raw["permitted_actions"]),
            )
            if raw["planner_hold"]:
                packet = hold_packet(
                    request,
                    "INSUFFICIENT_EVIDENCE",
                    model_id="stage3-development-fixture",
                )
            else:
                packet = ActionPacket(
                    schema_version="1.0",
                    case_id=request.case_id,
                    decision_time=request.decision_time,
                    root_cause=request.predicted_cause,
                    action_code=request.candidate_action,
                    evidence_ids=tuple(raw["evidence_ids"]),
                    confidence=0.95,
                    abstain=False,
                    reason_code="SUPPORTED_ACTION",
                    model_id="stage3-development-fixture",
                    prompt_version="finops_action_packet_v1",
                )
            hits = [
                RetrievalHit(
                    document_id=card_id,
                    source_type="POLICY_CARD",
                    score=float(len(raw["hit_ids"]) - index),
                    rank=index + 1,
                    allowed_actions=cards_by_id[card_id].allowed_actions,
                    checksum=cards_by_id[card_id].checksum,
                )
                for index, card_id in enumerate(raw["hit_ids"])
            ]
            context = PolicyContext(
                requested_target=raw["requested_target"],
                external_execution_requested=raw["external_execution_requested"],
                review_capacity_available=raw["review_capacity_available"],
                claim_boundary=CLAIM_BOUNDARY,
            )
            state = run_direct_orchestrator(request, packet, hits, cards, context)
            initial_route = state.route
            if raw["review_decision"] is not None:
                if state.review_request is None:
                    raise RuntimeError(f"{scenario_id} did not produce a review request")
                state = resume_human_review(
                    state,
                    review_token=state.review_request.review_token,
                    reviewer_id="stage3-development-reviewer",
                    decision=raw["review_decision"],
                )
            expectation_match = (
                state.verification.status == raw["expected_verification"]
                and initial_route == raw["expected_initial_route"]
                and state.route == raw["expected_final_route"]
            )
            results.append(
                {
                    "scenario_id": scenario_id,
                    "expected_verification": raw["expected_verification"],
                    "expected_initial_route": raw["expected_initial_route"],
                    "expected_final_route": raw["expected_final_route"],
                    "initial_route": initial_route,
                    "expectation_match": expectation_match,
                    "state": state.model_dump(mode="json"),
                }
            )

        metrics = _metrics(results)
        atomic_json(
            staging / "evaluation_input_snapshot.json",
            {
                "schema_version": STAGE3_BENCHMARK_SCHEMA_VERSION,
                "development_only": True,
                "source_cards_file": source_cards_path.name,
                "source_cards_sha256": sha256_file(source_cards_path),
                "scenarios_file": scenarios_path.name,
                "scenarios_sha256": sha256_file(scenarios_path),
                "scenarios": scenarios_source,
            },
        )
        atomic_json(
            staging / "orchestration_scenario_results.json",
            {
                "schema_version": STAGE3_BENCHMARK_SCHEMA_VERSION,
                "development_only": True,
                "results": results,
            },
        )
        atomic_json(staging / "orchestration_metrics.json", metrics)
        atomic_json(staging / "artifact_manifest.json", artifact_manifest(staging))
        verify_artifact_manifest(staging)
        publish_verified_tree(staging, output_dir)
        verification = verify_stage3_artifact(output_dir)
        if verification["status"] != "PASS":
            raise ArtifactVerificationError("published Stage 3 artifact failed verification")
        return verification
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_stage3_artifact(artifact_dir: Path) -> dict[str, Any]:
    integrity = verify_artifact_manifest(artifact_dir)
    raw_results = require_json_object(
        strict_json_load(artifact_dir / "orchestration_scenario_results.json"),
        source="Stage 3 scenario results",
    )
    if set(raw_results) != {"schema_version", "development_only", "results"}:
        raise ArtifactVerificationError("Stage 3 results schema is incompatible")
    if (
        raw_results["schema_version"] != STAGE3_BENCHMARK_SCHEMA_VERSION
        or raw_results["development_only"] is not True
        or not isinstance(raw_results["results"], list)
        or not raw_results["results"]
    ):
        raise ArtifactVerificationError("Stage 3 results contract is invalid")

    normalized: list[dict[str, Any]] = []
    for item in raw_results["results"]:
        if not isinstance(item, dict):
            raise ArtifactVerificationError("Stage 3 result record must be an object")
        required = {
            "scenario_id",
            "expected_verification",
            "expected_initial_route",
            "expected_final_route",
            "initial_route",
            "expectation_match",
            "state",
        }
        if set(item) != required:
            raise ArtifactVerificationError("Stage 3 result record schema is incompatible")
        state = OrchestrationState.model_validate(item["state"])
        independently_matched = (
            state.verification.status == item["expected_verification"]
            and item["initial_route"] == item["expected_initial_route"]
            and state.route == item["expected_final_route"]
        )
        if independently_matched is not item["expectation_match"]:
            raise ArtifactVerificationError("Stage 3 expectation flag was not reproducible")
        if state.receipt is not None and (
            state.receipt.external_execution or state.receipt.target != "TRADEOPS_SIM"
        ):
            raise ArtifactVerificationError("Stage 3 receipt violated the simulation boundary")
        if not trace_contract_is_complete(state):
            raise ArtifactVerificationError("Stage 3 trace completeness did not recompute")
        if state.verification.status == "REJECTED" and state.route != "HOLD":
            raise ArtifactVerificationError("Stage 3 verifier rejection did not fail closed")
        normalized.append({**item, "state": state.model_dump(mode="json")})

    recomputed = _metrics(normalized)
    recorded = require_json_object(
        strict_json_load(artifact_dir / "orchestration_metrics.json"),
        source="Stage 3 metrics",
    )
    if recorded != recomputed:
        raise ArtifactVerificationError("Stage 3 metrics did not independently recompute")
    return {
        "schema_version": STAGE3_BENCHMARK_SCHEMA_VERSION,
        "status": recomputed["status"],
        "checked_files": integrity["checked"],
        "scenario_count": recomputed["scenario_count"],
        "trace_complete_rate": recomputed["trace_complete_rate"],
        "external_execution_count": recomputed["external_execution_count"],
        "verifier_rejection_not_held_count": recomputed["verifier_rejection_not_held_count"],
        "artifact_manifest_sha256": sha256_file(artifact_dir / "artifact_manifest.json"),
    }
