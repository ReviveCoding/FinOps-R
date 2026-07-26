"""Machine-readable evidence contracts for GitHub-hosted E4 qualification."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from finops_r.integrated_qualification_stage5 import verify_integrated_qualification
from finops_r.json_safety import strict_json_dumps
from finops_r.stage4_qualification import verify_stage4_qualification

E4_SCHEMA_VERSION = "1.0"
SUPPORTED_CELLS = frozenset(
    {
        ("Linux", "3.11"),
        ("Linux", "3.12"),
        ("Windows", "3.11"),
        ("Windows", "3.12"),
    }
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _require_text(environment: Mapping[str, str], key: str) -> str:
    value = environment.get(key, "").strip()
    if not value:
        raise ValueError(f"missing required GitHub Actions environment value: {key}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid E4 evidence JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"E4 evidence must be a JSON object: {path}")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hosted_cell_evidence(
    *,
    output_path: Path,
    environment: Mapping[str, str],
    stage4_dir: Path,
    stage5_dir: Path,
) -> dict[str, Any]:
    """Write one E4 matrix-cell record after all locked checks have passed."""

    if environment.get("GITHUB_ACTIONS") != "true":
        raise ValueError("E4 hosted evidence can only be emitted by GitHub Actions")

    commit = _require_text(environment, "GITHUB_SHA").lower()
    expected_commit = _require_text(environment, "FINOPS_EXPECTED_COMMIT").lower()
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("GITHUB_SHA must be a full 40-character Git commit")
    if commit != expected_commit:
        raise ValueError("checked-out commit does not match FINOPS_EXPECTED_COMMIT")

    runner_os = _require_text(environment, "RUNNER_OS")
    matrix_python = _require_text(environment, "FINOPS_MATRIX_PYTHON")
    if (runner_os, matrix_python) not in SUPPORTED_CELLS:
        raise ValueError(f"unsupported E4 matrix cell: {(runner_os, matrix_python)!r}")
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != matrix_python:
        raise ValueError(
            f"running Python {actual_python} does not match matrix Python {matrix_python}"
        )

    stage4 = verify_stage4_qualification(stage4_dir)
    stage5 = verify_integrated_qualification(stage5_dir)
    if stage4["status"] != "PASS":
        raise ValueError("Stage 4 qualification is not PASS")
    if stage5["integrated_qualification_status"] != "PASS":
        raise ValueError("Stage 5 integrated qualification is not PASS")
    if stage5["model_promotion_status"] != "HOLD":
        raise ValueError("locked model-promotion judgment changed")
    if stage5["production_champion"] != "C1":
        raise ValueError("locked production champion changed")

    record: dict[str, Any] = {
        "schema_version": E4_SCHEMA_VERSION,
        "provider": "github_actions",
        "repository": _require_text(environment, "GITHUB_REPOSITORY"),
        "commit": commit,
        "workflow": _require_text(environment, "GITHUB_WORKFLOW"),
        "workflow_ref": _require_text(environment, "GITHUB_WORKFLOW_REF"),
        "run_id": _require_text(environment, "GITHUB_RUN_ID"),
        "run_attempt": _require_text(environment, "GITHUB_RUN_ATTEMPT"),
        "runner": {
            "os": runner_os,
            "arch": _require_text(environment, "RUNNER_ARCH"),
            "python": matrix_python,
            "python_full": sys.version,
        },
        "checks": {
            "dependency_consistency": "PASS",
            "test_suite": "PASS",
            "stage4_orchestration": stage4["status"],
            "stage4_scenarios": stage4["scenario_count"],
            "stage4_langgraph": stage4["langgraph_status"],
            "stage5_integrated": stage5["integrated_qualification_status"],
            "stage5_gates": stage5["gate_count"],
            "model_promotion": stage5["model_promotion_status"],
            "production_champion": stage5["production_champion"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        strict_json_dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def verify_hosted_e4_evidence(
    evidence_dir: Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    """Verify a complete four-cell E4 evidence set for one immutable commit."""

    expected_commit = expected_commit.lower()
    if not _COMMIT_PATTERN.fullmatch(expected_commit):
        raise ValueError("expected_commit must be a full 40-character Git commit")

    paths = sorted(evidence_dir.rglob("e4-cell-*.json"))
    if len(paths) != len(SUPPORTED_CELLS):
        raise ValueError(f"expected 4 E4 cell files, found {len(paths)}")

    cells: set[tuple[str, str]] = set()
    repositories: set[str] = set()
    run_ids: set[str] = set()
    file_records: list[dict[str, str]] = []
    for path in paths:
        raw = _load_json(path)
        if raw.get("schema_version") != E4_SCHEMA_VERSION:
            raise ValueError(f"unsupported E4 schema in {path}")
        if raw.get("provider") != "github_actions":
            raise ValueError(f"non-hosted provider in {path}")
        if raw.get("commit") != expected_commit:
            raise ValueError(f"commit mismatch in {path}")

        runner = raw.get("runner")
        checks = raw.get("checks")
        if not isinstance(runner, dict) or not isinstance(checks, dict):
            raise ValueError(f"missing runner/checks object in {path}")
        cell = (str(runner.get("os")), str(runner.get("python")))
        if cell not in SUPPORTED_CELLS or cell in cells:
            raise ValueError(f"unexpected or duplicate matrix cell in {path}: {cell!r}")
        cells.add(cell)

        required_checks = {
            "dependency_consistency": "PASS",
            "test_suite": "PASS",
            "stage4_orchestration": "PASS",
            "stage4_scenarios": 9,
            "stage4_langgraph": "PASS",
            "stage5_integrated": "PASS",
            "stage5_gates": 12,
            "model_promotion": "HOLD",
            "production_champion": "C1",
        }
        for key, expected in required_checks.items():
            if checks.get(key) != expected:
                raise ValueError(f"failed or changed check {key!r} in {path}")

        repository = raw.get("repository")
        run_id = raw.get("run_id")
        if not isinstance(repository, str) or not repository:
            raise ValueError(f"missing repository in {path}")
        if not isinstance(run_id, str) or not run_id.isdigit():
            raise ValueError(f"invalid run_id in {path}")
        repositories.add(repository)
        run_ids.add(run_id)
        file_records.append({"file": path.name, "sha256": _sha256(path)})

    if cells != SUPPORTED_CELLS:
        raise ValueError("E4 evidence matrix is incomplete")
    if len(repositories) != 1 or len(run_ids) != 1:
        raise ValueError("E4 cells do not belong to one repository and workflow run")

    return {
        "schema_version": E4_SCHEMA_VERSION,
        "e4_status": "PASS",
        "hosted_ci_status": "PASS",
        "qualified_commit": expected_commit,
        "repository": next(iter(repositories)),
        "run_id": next(iter(run_ids)),
        "matrix_cells": [{"os": os_name, "python": python} for os_name, python in sorted(cells)],
        "cell_files": file_records,
        "model_promotion_status": "HOLD",
        "production_champion": "C1",
    }
