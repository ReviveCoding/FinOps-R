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
from .checkpointing_stage4 import ReleasedTraceError, SQLiteCheckpointStore
from .json_safety import require_json_object, strict_json_load
from .langgraph_stage4 import (
    build_langgraph_stage4,
    invoke_langgraph_stage4,
    resume_langgraph_stage4,
)
from .orchestration_stage4 import (
    DurableStage4Orchestrator,
    InjectedCrash,
    Stage4Inputs,
    Stage4State,
    default_stage4_inputs,
    run_stage4_reference,
)
from .planner import ActionPacket, PlannerRequest, hold_packet
from .policy_cards import compile_policy_cards, load_compiled_policy_cards
from .retrieval import RetrievalHit
from .verification_policy import CLAIM_BOUNDARY, PolicyContext
from .verifier_stage4 import load_frozen_verifier, verify_frozen_verifier_provenance

STAGE4_QUALIFICATION_SCHEMA_VERSION = "1.0"


def _load_locked_scenarios(path: Path) -> dict[str, Any]:
    raw = require_json_object(strict_json_load(path), source="Stage 4 locked scenarios")
    if set(raw) != {"schema_version", "locked", "scenarios"}:
        raise ValueError("Stage 4 qualification scenario schema is incompatible")
    if raw["schema_version"] != STAGE4_QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("Stage 4 qualification schema version is incompatible")
    if raw["locked"] is not True:
        raise ValueError("Stage 4 qualification scenarios must be locked")
    scenarios = raw["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Stage 4 qualification requires non-empty scenarios")
    identifiers = [item.get("scenario_id") for item in scenarios if isinstance(item, dict)]
    if len(identifiers) != len(scenarios) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Stage 4 qualification scenario IDs must be unique")
    return raw


def _scenario_inputs(
    raw: dict[str, Any],
    *,
    cards: list[Any],
    verifier_path: Path,
) -> Stage4Inputs:
    request = PlannerRequest(
        case_id=f"STAGE4-{raw['scenario_id']}",
        decision_time=datetime(2026, 2, 17, 20, tzinfo=UTC),
        predicted_cause=raw["predicted_cause"],
        candidate_action=raw["candidate_action"],
        permitted_actions=tuple(raw["permitted_actions"]),
    )
    if raw["planner_hold"]:
        packet = hold_packet(
            request,
            "INSUFFICIENT_EVIDENCE",
            model_id="stage4-locked-fixture",
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
            model_id="Qwen/Qwen3-4B",
            prompt_version="finops_action_packet_v1",
        )
    cards_by_id = {card.card_id: card for card in cards}
    hits = [
        RetrievalHit(
            document_id=card_id,
            source_type="POLICY_CARD",
            score=float(len(raw["hit_ids"]) - index + 1),
            rank=index + 1,
            allowed_actions=cards_by_id[card_id].allowed_actions,
            checksum=cards_by_id[card_id].checksum,
        )
        for index, card_id in enumerate(raw["hit_ids"])
    ]
    return default_stage4_inputs(
        request=request,
        packet=packet,
        hits=hits,
        cards=cards,
        context=PolicyContext(
            requested_target=raw["requested_target"],
            external_execution_requested=raw["external_execution_requested"],
            review_capacity_available=raw["review_capacity_available"],
            claim_boundary=CLAIM_BOUNDARY,
        ),
        verifier_path=verifier_path,
    )


def _without_effect_flag(state: Stage4State) -> Stage4State:
    return state.model_copy(update={"simulation_effect_recorded": False})


def _langgraph_parity(inputs: Stage4Inputs, expected: Stage4State) -> dict[str, Any]:
    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError:
        return {
            "status": "NOT_RUN",
            "reason": "LANGGRAPH_OPTIONAL_DEPENDENCY_UNAVAILABLE",
        }
    graph = build_langgraph_stage4(InMemorySaver())
    thread_id = f"langgraph-{inputs.request.case_id}"
    actual = invoke_langgraph_stage4(graph, inputs, thread_id=thread_id)
    if actual.direct_state.status == "WAITING_REVIEW":
        review = actual.direct_state.review_request
        if review is None:
            raise RuntimeError("LangGraph parity scenario lost its review request")
        expected_resolution = expected.direct_state.review_resolution
        if expected_resolution is None:
            return {
                "status": "PASS",
                "paused": True,
                "parity": actual == expected,
            }
        actual = resume_langgraph_stage4(
            graph,
            thread_id=thread_id,
            review_token=review.review_token,
            reviewer_id=expected_resolution.reviewer_id,
            decision=expected_resolution.decision,
        )
    return {
        "status": "PASS" if actual == expected else "FAIL",
        "paused": expected.direct_state.review_request is not None,
        "parity": actual == expected,
    }


def run_stage4_qualification(
    *,
    source_cards_path: Path,
    scenarios_path: Path,
    verifier_path: Path,
    output_dir: Path,
    require_langgraph: bool = True,
) -> dict[str, Any]:
    source = _load_locked_scenarios(scenarios_path)
    verifier, _ = load_frozen_verifier(verifier_path)
    verifier_provenance = verify_frozen_verifier_provenance(
        verifier,
        training_path=verifier_path.with_name("learned_verifier_training_stage4.json"),
        threshold_path=verifier_path.with_name("learned_verifier_threshold_stage4.json"),
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}.staging."))
    try:
        compile_policy_cards(source_cards_path, staging / "compiled")
        cards = load_compiled_policy_cards(staging / "compiled" / "policy_cards.json")
        results: list[dict[str, Any]] = []
        langgraph_records: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory() as state_directory:
            state_root = Path(state_directory)
            for index, raw in enumerate(source["scenarios"]):
                inputs = _scenario_inputs(raw, cards=cards, verifier_path=verifier_path)
                reference = run_stage4_reference(inputs)
                initial_route = reference.effective_route
                if raw["review_decision"] is not None:
                    review = reference.direct_state.review_request
                    if review is None:
                        raise RuntimeError("qualification review scenario did not pause")
                    from .orchestration_stage3 import resume_human_review
                    from .orchestration_stage4 import build_stage4_state

                    reference = build_stage4_state(
                        inputs,
                        resume_human_review(
                            reference.direct_state,
                            review_token=review.review_token,
                            reviewer_id="stage4-locked-reviewer",
                            decision=raw["review_decision"],
                        ),
                    )
                graph_record = _langgraph_parity(inputs, reference)
                graph_record["scenario_id"] = raw["scenario_id"]
                langgraph_records.append(graph_record)

                store = SQLiteCheckpointStore(state_root / f"scenario-{index}.sqlite3")
                durable = DurableStage4Orchestrator(store)
                actual = durable.start(f"thread-{index}", inputs)
                if raw["review_decision"] is not None:
                    review = actual.direct_state.review_request
                    if review is None:
                        raise RuntimeError("durable qualification scenario did not pause")
                    actual = durable.resume_review(
                        f"thread-{index}",
                        review_token=review.review_token,
                        reviewer_id="stage4-locked-reviewer",
                        decision=raw["review_decision"],
                    )
                parity = _without_effect_flag(actual) == reference
                expectation = (
                    initial_route == raw["expected_initial_route"]
                    and actual.effective_route == raw["expected_final_route"]
                    and actual.learned_verifier.flagged is raw["expected_learned_flag"]
                )
                effect_count = store.simulation_effect_count(actual.direct_state.idempotency_key)
                effect_valid = effect_count == (1 if actual.effective_route == "SIMULATE" else 0)
                release = None
                immutable = False
                if actual.direct_state.status == "COMPLETED":
                    release = durable.release(f"thread-{index}")
                    try:
                        store.append(
                            f"thread-{index}",
                            "FORBIDDEN_MUTATION",
                            {"value": True},
                        )
                    except ReleasedTraceError:
                        immutable = True
                results.append(
                    {
                        "scenario_id": raw["scenario_id"],
                        "initial_route": initial_route,
                        "final_route": actual.effective_route,
                        "reference_parity": parity,
                        "expectation_match": expectation,
                        "effect_count": effect_count,
                        "effect_count_valid": effect_valid,
                        "released_trace_immutable": immutable,
                        "release": release,
                        "state": actual.model_dump(mode="json"),
                    }
                )

            crash_inputs = _scenario_inputs(
                source["scenarios"][0],
                cards=cards,
                verifier_path=verifier_path,
            )
            crash_store = SQLiteCheckpointStore(state_root / "crash.sqlite3")
            crash_runner = DurableStage4Orchestrator(crash_store)
            crash_observed = False
            try:
                crash_runner.start(
                    "crash-replay",
                    crash_inputs,
                    crash_at="after_effect",
                )
            except InjectedCrash:
                crash_observed = True
            recovered = DurableStage4Orchestrator(
                SQLiteCheckpointStore(state_root / "crash.sqlite3")
            ).recover("crash-replay")
            crash_gate = {
                "crash_observed": crash_observed,
                "recovered": recovered.direct_state.status == "COMPLETED",
                "effect_count": crash_store.simulation_effect_count(
                    recovered.direct_state.idempotency_key
                ),
            }

        langgraph_available = all(record["status"] != "NOT_RUN" for record in langgraph_records)
        langgraph_pass = langgraph_available and all(
            record["status"] == "PASS" and record["parity"] for record in langgraph_records
        )
        core_pass = all(
            item["reference_parity"]
            and item["expectation_match"]
            and item["effect_count_valid"]
            and item["released_trace_immutable"]
            for item in results
        ) and crash_gate == {
            "crash_observed": True,
            "recovered": True,
            "effect_count": 1,
        }
        status = (
            "PASS"
            if core_pass and (langgraph_pass or not require_langgraph)
            else "BLOCKED"
            if core_pass and require_langgraph and not langgraph_available
            else "FAIL"
        )
        metrics = {
            "schema_version": STAGE4_QUALIFICATION_SCHEMA_VERSION,
            "locked": True,
            "scenario_count": len(results),
            "reference_parity_rate": (
                sum(item["reference_parity"] for item in results) / len(results)
            ),
            "expectation_match_rate": (
                sum(item["expectation_match"] for item in results) / len(results)
            ),
            "duplicate_effect_count": sum(max(0, item["effect_count"] - 1) for item in results),
            "released_trace_immutability_rate": (
                sum(item["released_trace_immutable"] for item in results) / len(results)
            ),
            "external_execution_count": sum(
                bool(
                    item["state"]["direct_state"]["receipt"]
                    and item["state"]["direct_state"]["receipt"]["external_execution"]
                )
                for item in results
            ),
            "crash_replay": crash_gate,
            "langgraph_runtime_available": langgraph_available,
            "langgraph_parity_rate": (
                sum(record.get("parity", False) for record in langgraph_records)
                / len(langgraph_records)
            ),
            "require_langgraph": require_langgraph,
            "core_status": "PASS" if core_pass else "FAIL",
            "status": status,
        }
        atomic_json(
            staging / "qualification_input_snapshot.json",
            {
                "schema_version": STAGE4_QUALIFICATION_SCHEMA_VERSION,
                "locked": True,
                "source_cards_sha256": sha256_file(source_cards_path),
                "scenarios_sha256": sha256_file(scenarios_path),
                "verifier_artifact_sha256": sha256_file(verifier_path),
                "verifier_provenance": verifier_provenance,
                "scenarios": source,
            },
        )
        atomic_json(staging / "qualification_results.json", {"results": results})
        atomic_json(staging / "langgraph_parity.json", {"results": langgraph_records})
        atomic_json(staging / "qualification_metrics.json", metrics)
        shutil.rmtree(staging / "compiled")
        atomic_json(staging / "artifact_manifest.json", artifact_manifest(staging))
        verify_artifact_manifest(staging)
        publish_verified_tree(staging, output_dir)
        verified = verify_stage4_qualification(output_dir)
        if verified["status"] != status:
            raise ArtifactVerificationError("published Stage 4 qualification status changed")
        return verified
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_stage4_qualification(artifact_dir: Path) -> dict[str, Any]:
    integrity = verify_artifact_manifest(artifact_dir)
    snapshot = require_json_object(
        strict_json_load(artifact_dir / "qualification_input_snapshot.json"),
        source="Stage 4 qualification input snapshot",
    )
    metrics = require_json_object(
        strict_json_load(artifact_dir / "qualification_metrics.json"),
        source="Stage 4 qualification metrics",
    )
    results_root = require_json_object(
        strict_json_load(artifact_dir / "qualification_results.json"),
        source="Stage 4 qualification results",
    )
    graph_root = require_json_object(
        strict_json_load(artifact_dir / "langgraph_parity.json"),
        source="Stage 4 LangGraph parity results",
    )
    if snapshot.get("locked") is not True or metrics.get("locked") is not True:
        raise ArtifactVerificationError("Stage 4 qualification input is not locked")
    results = results_root.get("results")
    graph_results = graph_root.get("results")
    if not isinstance(results, list) or not results:
        raise ArtifactVerificationError("Stage 4 qualification results are empty")
    if not isinstance(graph_results, list) or len(graph_results) != len(results):
        raise ArtifactVerificationError("Stage 4 LangGraph result count is invalid")
    core_pass = all(
        item.get("reference_parity") is True
        and item.get("expectation_match") is True
        and item.get("effect_count_valid") is True
        and item.get("released_trace_immutable") is True
        for item in results
    )
    crash = metrics.get("crash_replay")
    if crash != {"crash_observed": True, "effect_count": 1, "recovered": True}:
        core_pass = False
    graph_available = all(item.get("status") != "NOT_RUN" for item in graph_results)
    graph_pass = graph_available and all(
        item.get("status") == "PASS" and item.get("parity") is True for item in graph_results
    )
    expected = (
        "PASS"
        if core_pass and (graph_pass or metrics.get("require_langgraph") is False)
        else "BLOCKED"
        if core_pass and metrics.get("require_langgraph") is True and not graph_available
        else "FAIL"
    )
    if metrics.get("status") != expected:
        raise ArtifactVerificationError("Stage 4 qualification status is not reproducible")
    if metrics.get("external_execution_count") != 0:
        raise ArtifactVerificationError("Stage 4 violated the simulation boundary")
    if metrics.get("duplicate_effect_count") != 0:
        raise ArtifactVerificationError("Stage 4 created a duplicate effect")
    return {
        "schema_version": STAGE4_QUALIFICATION_SCHEMA_VERSION,
        "status": expected,
        "core_status": "PASS" if core_pass else "FAIL",
        "langgraph_status": (
            "PASS" if graph_pass else "NOT_RUN" if not graph_available else "FAIL"
        ),
        "checked_files": integrity["checked"],
        "scenario_count": len(results),
        "qualification_metrics_sha256": sha256_file(artifact_dir / "qualification_metrics.json"),
    }
