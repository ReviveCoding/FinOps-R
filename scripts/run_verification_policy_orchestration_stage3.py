from __future__ import annotations

import argparse
from pathlib import Path

from finops_r.json_safety import strict_json_dumps
from finops_r.stage3_benchmark import run_stage3_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the FinOps-R Stage 3 deterministic verifier and direct orchestrator"
    )
    parser.add_argument(
        "--source-cards",
        type=Path,
        default=Path("data/policy_cards/source_cards.json"),
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("data/policy_cards/verification_policy_scenarios_stage3.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/components/verification_policy_orchestration_stage3_v1"),
    )
    args = parser.parse_args()
    result = run_stage3_benchmark(
        source_cards_path=args.source_cards,
        scenarios_path=args.scenarios,
        output_dir=args.output_dir,
    )
    print(strict_json_dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
