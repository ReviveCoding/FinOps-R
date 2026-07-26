from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import (
    ArtifactVerificationError,
    artifact_manifest,
    atomic_json,
    atomic_text,
    publish_verified_tree,
    sha256_file,
    verify_artifact_manifest,
)
from .json_safety import require_json_object, strict_json_load
from .pipeline import load_production_bundle
from .stage4_qualification import verify_stage4_qualification

STAGE5_SCHEMA_VERSION = "1.0"


def _load_contract(path: Path) -> dict[str, Any]:
    contract = require_json_object(
        strict_json_load(path), source="Stage 5 integrated qualification contract"
    )
    required = {
        "schema_version",
        "locked",
        "qualification_id",
        "claim_boundary",
        "model_contract",
        "orchestration_contract",
        "release_contract",
        "excluded_core_datasets",
    }
    if set(contract) != required:
        raise ArtifactVerificationError("Stage 5 contract schema is incompatible")
    if contract["schema_version"] != STAGE5_SCHEMA_VERSION or contract["locked"] is not True:
        raise ArtifactVerificationError("Stage 5 contract must be schema 1.0 and locked")
    if (
        not isinstance(contract["excluded_core_datasets"], list)
        or not contract["excluded_core_datasets"]
    ):
        raise ArtifactVerificationError("Stage 5 excluded dataset contract is empty")
    return contract


def _model_evidence(run_dir: Path) -> dict[str, Any]:
    integrity = verify_artifact_manifest(run_dir)
    bundle = load_production_bundle(run_dir)
    release = require_json_object(
        strict_json_load(run_dir / "release_decision.json"), source="model release decision"
    )
    run_manifest = require_json_object(
        strict_json_load(run_dir / "run_manifest.json"), source="model run manifest"
    )
    return {
        "run_id": run_dir.name,
        "project_version": run_manifest.get("project_version"),
        "artifact_files_checked": integrity["checked"],
        "artifact_manifest_sha256": sha256_file(run_dir / "artifact_manifest.json"),
        "release_decision_sha256": sha256_file(run_dir / "release_decision.json"),
        "release": release,
        "bundle": {
            "run_id": bundle["run_id"],
            "model_version": bundle["model_version"],
            "production_champion": bundle["production_champion"],
            "claim_boundary": bundle["claim_boundary"],
        },
    }


def _orchestration_evidence(artifact_dir: Path) -> dict[str, Any]:
    verification = verify_stage4_qualification(artifact_dir)
    metrics = require_json_object(
        strict_json_load(artifact_dir / "qualification_metrics.json"),
        source="Stage 4 qualification metrics",
    )
    return {
        "verification": verification,
        "metrics": metrics,
        "artifact_manifest_sha256": sha256_file(artifact_dir / "artifact_manifest.json"),
        "qualification_metrics_sha256": sha256_file(artifact_dir / "qualification_metrics.json"),
    }


def _upstream_evidence(path: Path) -> dict[str, Any]:
    upstream = require_json_object(
        strict_json_load(path), source="upstream source snapshot identity"
    )
    required = {
        "schema_version",
        "source_file",
        "source_bytes",
        "source_sha256",
        "zip_entry_count",
        "reconstruction_status",
        "role",
    }
    if set(upstream) != required:
        raise ArtifactVerificationError("upstream source snapshot schema is incompatible")
    checksum = upstream["source_sha256"]
    if (
        upstream["schema_version"] != "1.0"
        or upstream["reconstruction_status"] != "PASS"
        or upstream["role"] != "IMMUTABLE_STAGE4_UPSTREAM"
        or upstream["source_bytes"] != 2_884_587_891
        or upstream["zip_entry_count"] != 25_955
        or not isinstance(checksum, str)
        or len(checksum) != 64
    ):
        raise ArtifactVerificationError("upstream source snapshot identity is invalid")
    return {
        "identity": upstream,
        "identity_file_sha256": sha256_file(path),
    }


def evaluate_integrated_gates(
    contract: dict[str, Any],
    model: dict[str, Any],
    orchestration: dict[str, Any],
) -> dict[str, bool]:
    release = model["release"]
    bundle = model["bundle"]
    metrics = orchestration["metrics"]
    verification = orchestration["verification"]
    model_contract = contract["model_contract"]
    orchestration_contract = contract["orchestration_contract"]
    return {
        "model_artifact_integrity": model["artifact_files_checked"] > 0,
        "model_bundle_identity": (
            bundle["run_id"] == model["run_id"]
            and bundle["production_champion"] == release.get("production_champion")
        ),
        "model_claim_boundary": (
            release.get("claim_boundary") == "controlled TradeOps-Sim only"
            and bundle["claim_boundary"] == "controlled simulation"
        ),
        "model_release_posture": (
            release.get("status") == model_contract["expected_release_status"]
            and release.get("production_champion") == model_contract["expected_production_champion"]
        ),
        "unauthorized_action_zero": (
            release.get("unauthorized_action_count")
            == model_contract["required_unauthorized_action_count"]
        ),
        "stage4_verified": (
            verification["status"] == orchestration_contract["expected_status"]
            and verification["scenario_count"] == orchestration_contract["expected_scenario_count"]
        ),
        "stage4_core_pass": metrics.get("core_status") == "PASS",
        "langgraph_parity": (
            (not orchestration_contract["require_langgraph"])
            or (
                verification["langgraph_status"] == "PASS"
                and metrics.get("langgraph_runtime_available") is True
                and metrics.get("langgraph_parity_rate") == 1.0
            )
        ),
        "duplicate_effect_zero": (
            metrics.get("duplicate_effect_count")
            == orchestration_contract["required_duplicate_effect_count"]
        ),
        "external_execution_zero": (
            metrics.get("external_execution_count")
            == orchestration_contract["required_external_execution_count"]
        ),
        "released_trace_immutable": (
            metrics.get("released_trace_immutability_rate")
            == orchestration_contract["required_trace_immutability_rate"]
        ),
        "crash_replay_exactly_once": metrics.get("crash_replay")
        == {"crash_observed": True, "effect_count": 1, "recovered": True},
    }


def _render_report(
    *,
    model: dict[str, Any],
    orchestration: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    release = model["release"]
    metrics = orchestration["metrics"]
    failed = [name for name, passed in decision["gates"].items() if not passed]
    failed_text = ", ".join(failed) if failed else "none"
    return f"""# FinOps-R integrated qualification

- Package candidate: {__version__}
- Integrated qualification: {decision["integrated_qualification_status"]}
- Model promotion: {decision["model_promotion_status"]}
- Final release: {decision["final_release_status"]}
- Model run: {model["run_id"]}
- Production fallback: {release["production_champion"]}
- Stage 4 scenarios: {metrics["scenario_count"]}
- Duplicate simulation effects: {metrics["duplicate_effect_count"]}
- External executions: {metrics["external_execution_count"]}
- Failed integration gates: {failed_text}

The software and safety contracts can pass while model promotion remains HOLD. The locked
model evaluation did not establish C5 superiority, so C1 remains the production reference.
Hosted CI and Windows evidence are still required before a final 0.4.0 release.
"""


def run_integrated_qualification(
    *,
    contract_path: Path,
    model_run_dir: Path,
    stage4_artifact_dir: Path,
    upstream_identity_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    contract = _load_contract(contract_path)
    model = _model_evidence(model_run_dir.resolve())
    orchestration = _orchestration_evidence(stage4_artifact_dir.resolve())
    upstream = _upstream_evidence(upstream_identity_path.resolve())
    gates = evaluate_integrated_gates(contract, model, orchestration)
    integrated_status = "PASS" if all(gates.values()) else "FAIL"
    release_contract = contract["release_contract"]
    final_status = (
        release_contract["allowed_final_release_status_without_hosted_ci"]
        if release_contract["hosted_ci_required_for_final_release"]
        else integrated_status
    )
    decision = {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "qualification_id": contract["qualification_id"],
        "integrated_qualification_status": integrated_status,
        "model_promotion_status": model["release"]["status"],
        "production_champion": model["release"]["production_champion"],
        "local_evidence_level": release_contract["local_evidence_level"],
        "hosted_ci_status": "NOT_RUN",
        "final_release_status": final_status,
        "gates": gates,
        "claim_boundary": contract["claim_boundary"],
    }
    snapshot = {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "locked": True,
        "package_candidate_version": __version__,
        "contract_sha256": sha256_file(contract_path),
        "contract": contract,
        "model": model,
        "orchestration": orchestration,
        "upstream": upstream,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}.staging."))
    try:
        atomic_json(staging / "qualification_input_snapshot.json", snapshot)
        atomic_json(staging / "integrated_qualification_decision.json", decision)
        atomic_text(
            staging / "INTEGRATED_QUALIFICATION_REPORT.md",
            _render_report(model=model, orchestration=orchestration, decision=decision),
        )
        atomic_json(staging / "artifact_manifest.json", artifact_manifest(staging))
        verify_artifact_manifest(staging)
        publish_verified_tree(staging, output_dir)
        verified = verify_integrated_qualification(output_dir)
        if verified["integrated_qualification_status"] != integrated_status:
            raise ArtifactVerificationError("published Stage 5 qualification status changed")
        return verified
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_integrated_qualification(artifact_dir: Path) -> dict[str, Any]:
    integrity = verify_artifact_manifest(artifact_dir)
    snapshot = require_json_object(
        strict_json_load(artifact_dir / "qualification_input_snapshot.json"),
        source="Stage 5 qualification input snapshot",
    )
    decision = require_json_object(
        strict_json_load(artifact_dir / "integrated_qualification_decision.json"),
        source="Stage 5 integrated qualification decision",
    )
    if (
        snapshot.get("schema_version") != STAGE5_SCHEMA_VERSION
        or snapshot.get("locked") is not True
    ):
        raise ArtifactVerificationError("Stage 5 qualification snapshot is not locked")
    contract = snapshot.get("contract")
    model = snapshot.get("model")
    orchestration = snapshot.get("orchestration")
    if not isinstance(contract, dict):
        raise ArtifactVerificationError("Stage 5 qualification contract is missing")
    if not isinstance(model, dict):
        raise ArtifactVerificationError("Stage 5 qualification model evidence is missing")
    if not isinstance(orchestration, dict):
        raise ArtifactVerificationError("Stage 5 qualification snapshot is incomplete")
    expected_gates = evaluate_integrated_gates(contract, model, orchestration)
    expected_integrated = "PASS" if all(expected_gates.values()) else "FAIL"
    release_contract = contract["release_contract"]
    expected_final = (
        release_contract["allowed_final_release_status_without_hosted_ci"]
        if release_contract["hosted_ci_required_for_final_release"]
        else expected_integrated
    )
    if (
        decision.get("gates") != expected_gates
        or decision.get("integrated_qualification_status") != expected_integrated
        or decision.get("model_promotion_status") != model["release"]["status"]
        or decision.get("production_champion") != model["release"]["production_champion"]
        or decision.get("hosted_ci_status") != "NOT_RUN"
        or decision.get("final_release_status") != expected_final
    ):
        raise ArtifactVerificationError("Stage 5 qualification decision is not reproducible")
    return {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "integrated_qualification_status": expected_integrated,
        "model_promotion_status": decision["model_promotion_status"],
        "production_champion": decision["production_champion"],
        "hosted_ci_status": decision["hosted_ci_status"],
        "final_release_status": expected_final,
        "gate_count": len(expected_gates),
        "failed_gates": [name for name, passed in expected_gates.items() if not passed],
        "checked_files": integrity["checked"],
        "decision_sha256": sha256_file(artifact_dir / "integrated_qualification_decision.json"),
    }
