from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from finops_r import __version__
from finops_r.api import create_app
from finops_r.artifacts import (
    ArtifactVerificationError,
    artifact_manifest,
    atomic_json,
    verify_artifact_manifest,
)
from finops_r.contracts import CaseSnapshot
from finops_r.pipeline import (
    assess_request,
    load_production_bundle,
    regenerate_report,
    run_project,
)


class EndToEndTests(unittest.TestCase):
    def test_small_data_run_creates_product_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            config = {
                "project": "FinOps-R",
                "config_version": "test",
                "seed": 91,
                "n_cases": 650,
                "n_entities": 325,
                "output_root": "artifacts/runs",
                "review_capacity": 0.2,
                "human_correction_probability": 0.85,
                "review_cost": 0.1,
                "hold_cost_fraction": 0.1,
                "hold_fixed_cost": 0.1,
                "critical_false_greenlight_limit": 0.5,
                "evidence_precision_limit": 0.5,
                "min_relative_harm_improvement": -1.0,
                "temperature_bounds": [0.35, 4.0],
                "bootstrap_resamples": 100,
                "specialty_resamples": 100,
                "ensemble": {
                    "hist_max_iter": 18,
                    "hist_learning_rate": 0.1,
                    "hist_max_leaf_nodes": 15,
                    "extra_trees_estimators": 16,
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
            config_path = config_dir / "test.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            run_dir = run_project(config_path, root / "runs")
            for relative in (
                "metrics/metrics.json",
                "metrics/candidate_comparison.csv",
                "REPORT.html",
                "project_creation_handoff.json",
                "artifact_integrity.json",
                "models/production_bundle.joblib",
            ):
                self.assertTrue((run_dir / relative).exists(), relative)
            fixture = Path(__file__).parents[1] / "data" / "fixtures" / "sample_case.json"
            assessment = assess_request(run_dir, fixture)
            self.assertIn(assessment["route"], {"SIMULATE", "REVIEW", "HOLD"})
            self.assertEqual(assessment["claim_boundary"], "controlled simulation")
            self.assertEqual(verify_artifact_manifest(run_dir)["status"], "PASS")
            self.assertFalse(
                any(
                    part.startswith(".")
                    for path in run_dir.rglob("*")
                    for part in path.relative_to(run_dir).parts
                )
            )
            release = json.loads((run_dir / "release_decision.json").read_text(encoding="utf-8"))
            self.assertIn("stress_manual_intervention_capacity", release["gates"])
            policy = json.loads(
                (run_dir / "models" / "decision_policy.json").read_text(encoding="utf-8")
            )
            self.assertEqual(policy["fit_split"], "policy")
            self.assertEqual(set(policy["minimum_review_scores"]), {f"C{i}" for i in range(6)})

            app = create_app(run_dir)
            self.assertEqual(app.version, __version__)
            health_endpoint = next(
                route.endpoint for route in app.routes if route.path == "/health"
            )
            assess_endpoint = next(
                route.endpoint for route in app.routes if route.path == "/v1/cases/assess"
            )
            self.assertEqual(health_endpoint()["artifact_trust"], "verified_local_only")
            snapshot = CaseSnapshot.model_validate_json(fixture.read_text(encoding="utf-8"))
            api_assessment = assess_endpoint(snapshot)
            self.assertEqual(api_assessment.model_version, assessment["model_version"])
            self.assertEqual(api_assessment.route, assessment["route"])

            source_manifest = (run_dir / "artifact_manifest.json").read_bytes()
            refreshed_run, refreshed_figures = regenerate_report(run_dir)
            self.assertNotEqual(refreshed_run, run_dir)
            self.assertGreaterEqual(len(refreshed_figures), 14)
            self.assertEqual((run_dir / "artifact_manifest.json").read_bytes(), source_manifest)
            self.assertEqual(verify_artifact_manifest(refreshed_run)["status"], "PASS")
            self.assertEqual(
                load_production_bundle(refreshed_run)["model_version"], assessment["model_version"]
            )

            latest = root / "releases" / "latest.json"
            latest_record = json.loads(latest.read_text(encoding="utf-8"))
            self.assertEqual(latest_record["run_dir"], f"runs/{run_dir.name}")
            self.assertFalse((root / "artifacts").exists())
            latest_before = latest.read_bytes()
            interrupted_root = root / "interrupted-runs"
            with patch(
                "finops_r.pipeline.generate_cases", side_effect=RuntimeError("injected failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    run_project(config_path, interrupted_root)
            self.assertEqual(latest.read_bytes(), latest_before)
            self.assertEqual(
                [path for path in interrupted_root.iterdir() if not path.name.startswith(".")], []
            )
            self.assertEqual(list(Path(tempfile.gettempdir()).glob(f".{run_dir.name}*")), [])

            bundle_path = run_dir / "models" / "production_bundle.joblib"
            bundle = bundle_path.read_bytes()

            def replace_bundle_after_verification(path: Path) -> dict[str, object]:
                result = verify_artifact_manifest(path)
                bundle_path.write_bytes(b"replaced after verification")
                return result

            with (
                patch(
                    "finops_r.pipeline.verify_artifact_manifest",
                    side_effect=replace_bundle_after_verification,
                ),
                self.assertRaisesRegex(ArtifactVerificationError, "changed after"),
            ):
                load_production_bundle(run_dir)
            bundle_path.write_bytes(bundle)

            semantic_bundle = joblib.load(bundle_path)
            semantic_bundle["use_rsop"] = "false"
            joblib.dump(semantic_bundle, bundle_path, compress=("zlib", 3))
            atomic_json(run_dir / "artifact_manifest.json", artifact_manifest(run_dir))
            atomic_json(run_dir / "artifact_integrity.json", verify_artifact_manifest(run_dir))
            with self.assertRaisesRegex(ArtifactVerificationError, "feature flags"):
                load_production_bundle(run_dir)

            bundle_path.write_bytes(bundle)
            atomic_json(run_dir / "artifact_manifest.json", artifact_manifest(run_dir))
            atomic_json(run_dir / "artifact_integrity.json", verify_artifact_manifest(run_dir))
            bundle_path.write_bytes(bundle[:-1] + bytes([bundle[-1] ^ 0x01]))
            with self.assertRaises(ArtifactVerificationError):
                load_production_bundle(run_dir)


if __name__ == "__main__":
    unittest.main()
