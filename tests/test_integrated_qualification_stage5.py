from __future__ import annotations

from pathlib import Path

import pytest

from finops_r.artifacts import ArtifactVerificationError, artifact_manifest, atomic_json
from finops_r.integrated_qualification_stage5 import (
    evaluate_integrated_gates,
    verify_integrated_qualification,
)


def _contract() -> dict:
    return {
        "schema_version": "1.0",
        "locked": True,
        "qualification_id": "test",
        "claim_boundary": "controlled TradeOps-Sim and simulation-only orchestration",
        "model_contract": {
            "expected_release_status": "HOLD",
            "expected_production_champion": "C1",
            "required_unauthorized_action_count": 0,
        },
        "orchestration_contract": {
            "expected_status": "PASS",
            "expected_scenario_count": 9,
            "required_duplicate_effect_count": 0,
            "required_external_execution_count": 0,
            "required_trace_immutability_rate": 1.0,
            "require_langgraph": True,
        },
        "release_contract": {
            "local_evidence_level": "E3",
            "hosted_ci_required_for_final_release": True,
            "allowed_final_release_status_without_hosted_ci": "BLOCKED_EXTERNAL_CI",
        },
        "excluded_core_datasets": ["FUNSD"],
    }


def _model() -> dict:
    return {
        "run_id": "run-1",
        "artifact_files_checked": 59,
        "release": {
            "status": "HOLD",
            "production_champion": "C1",
            "claim_boundary": "controlled TradeOps-Sim only",
            "unauthorized_action_count": 0,
        },
        "bundle": {
            "run_id": "run-1",
            "production_champion": "C1",
            "claim_boundary": "controlled simulation",
        },
    }


def _orchestration() -> dict:
    return {
        "verification": {
            "status": "PASS",
            "scenario_count": 9,
            "langgraph_status": "PASS",
        },
        "metrics": {
            "core_status": "PASS",
            "langgraph_runtime_available": True,
            "langgraph_parity_rate": 1.0,
            "duplicate_effect_count": 0,
            "external_execution_count": 0,
            "released_trace_immutability_rate": 1.0,
            "crash_replay": {
                "crash_observed": True,
                "effect_count": 1,
                "recovered": True,
            },
        },
    }


def test_integrated_gate_contract_passes_without_promoting_model() -> None:
    gates = evaluate_integrated_gates(_contract(), _model(), _orchestration())
    assert gates
    assert all(gates.values())


@pytest.mark.parametrize(
    ("path", "value", "gate"),
    [
        (("release", "unauthorized_action_count"), 1, "unauthorized_action_zero"),
        (("bundle", "production_champion"), "C5", "model_bundle_identity"),
    ],
)
def test_model_safety_regression_fails_gate(
    path: tuple[str, str], value: object, gate: str
) -> None:
    model = _model()
    model[path[0]][path[1]] = value
    assert evaluate_integrated_gates(_contract(), model, _orchestration())[gate] is False


def test_orchestration_duplicate_effect_fails_gate() -> None:
    orchestration = _orchestration()
    orchestration["metrics"]["duplicate_effect_count"] = 1
    gates = evaluate_integrated_gates(_contract(), _model(), orchestration)
    assert gates["duplicate_effect_zero"] is False


def test_independent_verifier_rejects_tampered_decision(tmp_path: Path) -> None:
    root = tmp_path / "qualification"
    root.mkdir()
    contract = _contract()
    model = _model()
    orchestration = _orchestration()
    snapshot = {
        "schema_version": "1.0",
        "locked": True,
        "contract": contract,
        "model": model,
        "orchestration": orchestration,
    }
    gates = evaluate_integrated_gates(contract, model, orchestration)
    decision = {
        "gates": gates,
        "integrated_qualification_status": "FAIL",
        "model_promotion_status": "HOLD",
        "production_champion": "C1",
        "hosted_ci_status": "NOT_RUN",
        "final_release_status": "BLOCKED_EXTERNAL_CI",
    }
    atomic_json(root / "qualification_input_snapshot.json", snapshot)
    atomic_json(root / "integrated_qualification_decision.json", decision)
    (root / "INTEGRATED_QUALIFICATION_REPORT.md").write_text("test\n", encoding="utf-8")
    atomic_json(root / "artifact_manifest.json", artifact_manifest(root))
    with pytest.raises(ArtifactVerificationError, match="decision is not reproducible"):
        verify_integrated_qualification(root)


def test_independent_verifier_detects_file_tampering(tmp_path: Path) -> None:
    root = tmp_path / "qualification"
    root.mkdir()
    contract = _contract()
    model = _model()
    orchestration = _orchestration()
    gates = evaluate_integrated_gates(contract, model, orchestration)
    atomic_json(
        root / "qualification_input_snapshot.json",
        {
            "schema_version": "1.0",
            "locked": True,
            "contract": contract,
            "model": model,
            "orchestration": orchestration,
        },
    )
    atomic_json(
        root / "integrated_qualification_decision.json",
        {
            "gates": gates,
            "integrated_qualification_status": "PASS",
            "model_promotion_status": "HOLD",
            "production_champion": "C1",
            "hosted_ci_status": "NOT_RUN",
            "final_release_status": "BLOCKED_EXTERNAL_CI",
        },
    )
    report = root / "INTEGRATED_QUALIFICATION_REPORT.md"
    report.write_text("original\n", encoding="utf-8")
    atomic_json(root / "artifact_manifest.json", artifact_manifest(root))
    report.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError):
        verify_integrated_qualification(root)
