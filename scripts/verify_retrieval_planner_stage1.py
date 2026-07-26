from __future__ import annotations

import argparse
from pathlib import Path

from finops_r.json_safety import (
    require_json_object,
    strict_json_dumps,
    strict_json_load,
)
from finops_r.policy_cards import load_compiled_policy_cards


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify FinOps-R retrieval/planner stage 1")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/components/retrieval_planner_stage1"),
    )
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    cards = load_compiled_policy_cards(root / "policy_cards.json")
    benchmark = require_json_object(
        strict_json_load(root / "retrieval_benchmark.json"),
        source="retrieval benchmark",
    )
    manifest = require_json_object(
        strict_json_load(root / "retrieval_index_manifest.json"),
        source="retrieval manifest",
    )
    failures: list[str] = []
    if len(cards) < 7:
        failures.append("POLICY_CARD_COUNT")
    if benchmark.get("query_count") != 7:
        failures.append("QUERY_COUNT")
    if float(benchmark.get("mean_recall_at_k", 0.0)) < 1.0:
        failures.append("RECALL_AT_K")
    if float(benchmark.get("mean_reciprocal_rank", 0.0)) < 0.90:
        failures.append("MEAN_RECIPROCAL_RANK")
    if benchmark.get("future_evidence_count") != 0:
        failures.append("FUTURE_EVIDENCE")
    if manifest.get("effective_time_filter") is None:
        failures.append("EFFECTIVE_TIME_FILTER")
    result = {
        "schema_version": "1.0",
        "policy_card_count": len(cards),
        "mean_recall_at_k": benchmark.get("mean_recall_at_k"),
        "mean_reciprocal_rank": benchmark.get("mean_reciprocal_rank"),
        "future_evidence_count": benchmark.get("future_evidence_count"),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    print(strict_json_dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
