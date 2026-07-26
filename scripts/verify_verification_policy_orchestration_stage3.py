from __future__ import annotations

import argparse
from pathlib import Path

from finops_r.json_safety import strict_json_dumps
from finops_r.stage3_benchmark import verify_stage3_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the FinOps-R Stage 3 artifact")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/components/verification_policy_orchestration_stage3_v1"),
    )
    args = parser.parse_args()
    result = verify_stage3_artifact(args.artifact_dir)
    print(strict_json_dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
