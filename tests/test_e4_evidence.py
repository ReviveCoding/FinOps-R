from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from finops_r.e4_evidence import (
    SUPPORTED_CELLS,
    verify_hosted_e4_evidence,
    write_hosted_cell_evidence,
)


class HostedE4EvidenceTests(unittest.TestCase):
    commit = "a" * 40
    repository_root = Path(__file__).parents[1]

    @staticmethod
    def _record(os_name: str, python: str) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "provider": "github_actions",
            "repository": "owner/finops-r",
            "commit": HostedE4EvidenceTests.commit,
            "workflow": "ci",
            "workflow_ref": "owner/finops-r/.github/workflows/ci.yml@refs/heads/main",
            "run_id": "123456",
            "run_attempt": "1",
            "runner": {
                "os": os_name,
                "arch": "X64",
                "python": python,
                "python_full": f"{python}.0 test",
            },
            "checks": {
                "dependency_consistency": "PASS",
                "test_suite": "PASS",
                "stage4_orchestration": "PASS",
                "stage4_scenarios": 9,
                "stage4_langgraph": "PASS",
                "stage5_integrated": "PASS",
                "stage5_gates": 12,
                "model_promotion": "HOLD",
                "production_champion": "C1",
            },
        }

    def _write_matrix(self, root: Path) -> None:
        for os_name, python in SUPPORTED_CELLS:
            path = root / f"e4-cell-{os_name.lower()}-py{python}.json"
            path.write_text(
                json.dumps(self._record(os_name, python), sort_keys=True),
                encoding="utf-8",
            )

    def test_complete_locked_matrix_passes(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_matrix(root)
            result = verify_hosted_e4_evidence(root, expected_commit=self.commit)
        self.assertEqual(result["e4_status"], "PASS")
        self.assertEqual(result["hosted_ci_status"], "PASS")
        self.assertEqual(result["model_promotion_status"], "HOLD")
        self.assertEqual(result["production_champion"], "C1")
        self.assertEqual(len(result["matrix_cells"]), 4)

    def test_commit_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_matrix(root)
            with self.assertRaisesRegex(ValueError, "commit mismatch"):
                verify_hosted_e4_evidence(root, expected_commit="b" * 40)

    def test_incomplete_matrix_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_matrix(root)
            next(root.glob("e4-cell-*.json")).unlink()
            with self.assertRaisesRegex(ValueError, "expected 4"):
                verify_hosted_e4_evidence(root, expected_commit=self.commit)

    def test_model_judgment_change_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_matrix(root)
            path = next(root.glob("e4-cell-*.json"))
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["checks"]["model_promotion"] = "PASS"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_promotion"):
                verify_hosted_e4_evidence(root, expected_commit=self.commit)

    def _hosted_environment(self) -> dict[str, str]:
        return {
            "GITHUB_ACTIONS": "true",
            "GITHUB_SHA": self.commit,
            "FINOPS_EXPECTED_COMMIT": self.commit,
            "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64",
            "FINOPS_MATRIX_PYTHON": "3.12",
            "GITHUB_REPOSITORY": "owner/finops-r",
            "GITHUB_WORKFLOW": "ci",
            "GITHUB_WORKFLOW_REF": ("owner/finops-r/.github/workflows/ci.yml@refs/heads/main"),
            "GITHUB_RUN_ID": "123456",
            "GITHUB_RUN_ATTEMPT": "1",
        }

    def test_hosted_writer_reverifies_locked_stage4_and_stage5(self) -> None:
        with TemporaryDirectory() as temp:
            output = Path(temp) / "e4-cell-linux-py3.12.json"
            result = write_hosted_cell_evidence(
                output_path=output,
                environment=self._hosted_environment(),
                stage4_dir=(
                    self.repository_root / "artifacts/qualification/orchestration_stage4_v1r2"
                ),
                stage5_dir=(
                    self.repository_root / "artifacts/qualification/integrated_stage5_v1r1"
                ),
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, persisted)
        self.assertEqual(result["checks"]["stage4_orchestration"], "PASS")
        self.assertEqual(result["checks"]["stage5_integrated"], "PASS")
        self.assertEqual(result["checks"]["model_promotion"], "HOLD")

    def test_writer_rejects_non_hosted_execution(self) -> None:
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "GitHub Actions"):
                write_hosted_cell_evidence(
                    output_path=Path(temp) / "evidence.json",
                    environment={},
                    stage4_dir=Path("unused"),
                    stage5_dir=Path("unused"),
                )

    def test_writer_rejects_checkout_commit_mismatch(self) -> None:
        environment = self._hosted_environment()
        environment["FINOPS_EXPECTED_COMMIT"] = "b" * 40
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "does not match"):
                write_hosted_cell_evidence(
                    output_path=Path(temp) / "evidence.json",
                    environment=environment,
                    stage4_dir=Path("unused"),
                    stage5_dir=Path("unused"),
                )


if __name__ == "__main__":
    unittest.main()
