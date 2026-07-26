from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from pydantic import ValidationError

from finops_r import __version__
from finops_r.artifacts import ArtifactVerificationError
from finops_r.contracts import Assessment, CaseSnapshot, ProjectConfig
from finops_r.pipeline import _run_identity, runtime_contract, verify_runtime_contract


class ContractTests(unittest.TestCase):
    def test_package_and_project_versions_cannot_drift(self) -> None:
        project = Path(__file__).parents[1] / "pyproject.toml"
        metadata = tomllib.loads(project.read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["version"], __version__)

    def test_timezone_is_required(self) -> None:
        payload = {
            "case_id": "T1",
            "entity_id": "E1",
            "decision_time": "2027-01-01T12:00:00",
            "state": "EXECUTED",
            "product": "EQUITY",
            "route": "DTC",
            "currency": "USD",
            "side": "BUY",
            "time_in_state_min": 1,
            "case_age_min": 1,
            "deadline_slack_min": 10,
            "event_count": 1,
            "artifact_count": 1,
            "missing_doc_count": 0,
        }
        with self.assertRaises(ValidationError):
            CaseSnapshot.model_validate(payload)

    def test_non_finite_value_is_rejected(self) -> None:
        payload = {
            "case_id": "T1",
            "entity_id": "E1",
            "decision_time": "2027-01-01T12:00:00Z",
            "state": "EXECUTED",
            "product": "EQUITY",
            "route": "DTC",
            "currency": "USD",
            "side": "BUY",
            "time_in_state_min": float("inf"),
            "case_age_min": 1,
            "deadline_slack_min": 10,
            "event_count": 1,
            "artifact_count": 1,
            "missing_doc_count": 0,
        }
        with self.assertRaises(ValidationError):
            CaseSnapshot.model_validate(payload)

    def test_fractional_binary_observable_is_rejected(self) -> None:
        payload = {
            "case_id": "T1",
            "entity_id": "E1",
            "decision_time": "2027-01-01T12:00:00Z",
            "state": "EXECUTED",
            "product": "EQUITY",
            "route": "DTC",
            "currency": "USD",
            "side": "BUY",
            "time_in_state_min": 1,
            "case_age_min": 1,
            "deadline_slack_min": 10,
            "event_count": 1,
            "artifact_count": 1,
            "missing_doc_count": 0,
            "allocation_present": 0.5,
        }
        with self.assertRaisesRegex(ValidationError, "binary"):
            CaseSnapshot.model_validate(payload)

    def test_boolean_values_are_not_coerced_into_numeric_contract_fields(self) -> None:
        case_payload = {
            "case_id": "T1",
            "entity_id": "E1",
            "decision_time": "2027-01-01T12:00:00Z",
            "state": "EXECUTED",
            "product": "EQUITY",
            "route": "DTC",
            "currency": "USD",
            "side": "BUY",
            "time_in_state_min": 1,
            "case_age_min": 1,
            "deadline_slack_min": 10,
            "event_count": True,
            "artifact_count": 1,
            "missing_doc_count": 0,
        }
        with self.assertRaisesRegex(ValidationError, "boolean"):
            CaseSnapshot.model_validate(case_payload)

        config_path = Path(__file__).parents[1] / "configs" / "smoke.json"
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
        for mutation in (
            {"seed": True},
            {"review_capacity": True},
            {"temperature_bounds": [True, 4.0]},
            {"ensemble": {**config["ensemble"], "hist_max_iter": True}},
        ):
            with (
                self.subTest(mutation=mutation),
                self.assertRaises(ValidationError),
            ):
                ProjectConfig.model_validate({**config, **mutation})

    def test_output_root_parent_traversal_is_rejected(self) -> None:
        payload = {
            "project": "FinOps-R",
            "config_version": "test",
            "seed": 1,
            "n_cases": 200,
            "n_entities": 100,
            "output_root": "../outside",
            "review_capacity": 0.2,
            "human_correction_probability": 0.85,
            "review_cost": 0.1,
            "hold_cost_fraction": 0.1,
            "hold_fixed_cost": 0.1,
            "critical_false_greenlight_limit": 0.5,
            "evidence_precision_limit": 0.5,
            "min_relative_harm_improvement": 0.0,
            "temperature_bounds": [0.35, 4.0],
            "ensemble": {
                "hist_max_iter": 10,
                "hist_learning_rate": 0.1,
                "hist_max_leaf_nodes": 15,
                "extra_trees_estimators": 10,
                "extra_trees_min_samples_leaf": 2,
                "weight_hist": 0.6,
                "weight_extra": 0.4,
            },
            "rsop": {"min_reference_count": 4, "epsilon": 0.001},
            "action_envelope": {
                "cumulative_probability": 0.82,
                "max_candidates": 3,
                "rule_expand_threshold": 0.65,
            },
        }
        with self.assertRaises(ValidationError):
            ProjectConfig.model_validate(payload)

    def test_windows_absolute_and_non_normalized_output_roots_are_rejected(self) -> None:
        base = {
            "project": "FinOps-R",
            "config_version": "test",
            "seed": 1,
            "n_cases": 200,
            "n_entities": 100,
            "review_capacity": 0.2,
            "human_correction_probability": 0.85,
            "review_cost": 0.1,
            "hold_cost_fraction": 0.1,
            "hold_fixed_cost": 0.1,
            "critical_false_greenlight_limit": 0.5,
            "evidence_precision_limit": 0.5,
            "min_relative_harm_improvement": 0.0,
            "temperature_bounds": [0.35, 4.0],
            "ensemble": {
                "hist_max_iter": 10,
                "hist_learning_rate": 0.1,
                "hist_max_leaf_nodes": 15,
                "extra_trees_estimators": 10,
                "extra_trees_min_samples_leaf": 2,
                "weight_hist": 0.6,
                "weight_extra": 0.4,
            },
            "rsop": {"min_reference_count": 4, "epsilon": 0.001},
            "action_envelope": {
                "cumulative_probability": 0.82,
                "max_candidates": 3,
                "rule_expand_threshold": 0.65,
            },
        }
        for output_root in (
            "C:\\outside",
            "artifacts//runs",
            ".",
            "CON",
            "artifacts/com1/runs",
            "artifacts/trailing./runs",
            "x" * 256,
        ):
            with self.subTest(output_root=output_root), self.assertRaises(ValidationError):
                ProjectConfig.model_validate({**base, "output_root": output_root})

    def test_generator_dimensions_and_seed_are_validated_before_execution(self) -> None:
        base = {
            "project": "FinOps-R",
            "config_version": "test",
            "seed": 1,
            "n_cases": 200,
            "n_entities": 100,
            "output_root": "artifacts/runs",
            "review_capacity": 0.2,
            "human_correction_probability": 0.85,
            "review_cost": 0.1,
            "hold_cost_fraction": 0.1,
            "hold_fixed_cost": 0.1,
            "critical_false_greenlight_limit": 0.5,
            "evidence_precision_limit": 0.5,
            "min_relative_harm_improvement": 0.0,
            "temperature_bounds": [0.35, 4.0],
            "ensemble": {
                "hist_max_iter": 10,
                "hist_learning_rate": 0.1,
                "hist_max_leaf_nodes": 15,
                "extra_trees_estimators": 10,
                "extra_trees_min_samples_leaf": 2,
                "weight_hist": 0.6,
                "weight_extra": 0.4,
            },
            "rsop": {"min_reference_count": 4, "epsilon": 0.001},
            "action_envelope": {
                "cumulative_probability": 0.82,
                "max_candidates": 3,
                "rule_expand_threshold": 0.65,
            },
        }
        for override in ({"seed": -1}, {"n_entities": 201}):
            with self.subTest(override=override), self.assertRaises(ValidationError):
                ProjectConfig.model_validate({**base, **override})

    def test_production_fallback_assessment_contract_accepts_not_applicable(self) -> None:
        assessment = {
            "case_id": "T1",
            "predicted_cause": "NO_EXCEPTION",
            "proposed_action": "NO_ACTION_MONITOR",
            "confidence": 0.9,
            "calibrated_error_risk": 0.1,
            "route": "SIMULATE",
            "reason_codes": [],
            "evidence_ids": [],
            "policy_status": "NOT_APPLICABLE",
            "verifier_status": "NOT_APPLICABLE",
            "model_version": "C1-0.1.0",
            "run_id": "run",
            "claim_boundary": "controlled simulation",
        }
        parsed = Assessment.model_validate(assessment)
        self.assertEqual(parsed.policy_status, "NOT_APPLICABLE")

    def test_incompatible_model_runtime_is_rejected_before_deserialization(self) -> None:
        contract = runtime_contract()
        contract["packages"]["scikit-learn"] = "0.0.0-incompatible"
        with self.assertRaisesRegex(ArtifactVerificationError, "package runtime mismatch"):
            verify_runtime_contract(contract)

    def test_run_identity_is_unique_for_concurrent_same_config(self) -> None:
        config = {"seed": 91}
        values = {_run_identity(config) for _ in range(100)}
        self.assertEqual(len(values), 100)
        for value in values:
            self.assertRegex(
                value,
                r"^finopsr_\d{8}T\d{6}Z_[0-9a-f]{10}_[0-9a-f]{32}$",
            )


if __name__ == "__main__":
    unittest.main()
