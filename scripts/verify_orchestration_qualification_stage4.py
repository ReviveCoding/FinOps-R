from __future__ import annotations

import argparse
from pathlib import Path

from finops_r.json_safety import strict_json_dumps
from finops_r.stage4_qualification import verify_stage4_qualification


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify the FinOps-R Stage 4 qualification artifact"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/qualification/orchestration_stage4_v1"),
    )
    args = parser.parse_args()
    result = verify_stage4_qualification(args.artifact_dir)
    print(strict_json_dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
