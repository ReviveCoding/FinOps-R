from __future__ import annotations

import argparse
from pathlib import Path

from finops_r.json_safety import strict_json_dumps
from finops_r.policy_cards import compile_policy_cards
from finops_r.retrieval import build_policy_bm25_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and benchmark FinOps-R policy retrieval")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/policy_cards/source_cards.json"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/policy_cards/retrieval_queries.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/components/retrieval_planner_stage1"),
    )
    args = parser.parse_args()
    policy = compile_policy_cards(args.source, args.output)
    retrieval = build_policy_bm25_artifacts(
        args.output / "policy_cards.json", args.queries, args.output
    )
    print(
        strict_json_dumps(
            {"policy_cards": policy, "retrieval": retrieval, "status": "PASS"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
