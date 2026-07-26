from __future__ import annotations

import argparse
from pathlib import Path

from finops_r.json_safety import strict_json_dumps
from finops_r.stage4_qualification import run_stage4_qualification


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the locked FinOps-R Stage 4 orchestration qualification"
    )
    parser.add_argument(
        "--source-cards",
        type=Path,
        default=Path("data/policy_cards/source_cards.json"),
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("data/policy_cards/orchestration_qualification_stage4_locked.json"),
    )
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path("data/policy_cards/frozen_learned_verifier_stage4.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/qualification/orchestration_stage4_v1"),
    )
    parser.add_argument(
        "--allow-missing-langgraph",
        action="store_true",
        help="Qualify the durable core while reporting LangGraph parity as not run.",
    )
    args = parser.parse_args()
    result = run_stage4_qualification(
        source_cards_path=args.source_cards,
        scenarios_path=args.scenarios,
        verifier_path=args.verifier,
        output_dir=args.output_dir,
        require_langgraph=not args.allow_missing_langgraph,
    )
    print(strict_json_dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
