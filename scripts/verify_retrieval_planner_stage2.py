from __future__ import annotations

import argparse
from pathlib import Path

from finops_r.json_safety import strict_json_dumps
from finops_r.stage2_verification import verify_stage2_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify FinOps-R retrieval/planner Stage 2")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/components/retrieval_planner_stage2"),
    )
    args = parser.parse_args()
    result = verify_stage2_artifact(args.artifact_dir)
    print(strict_json_dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
