from __future__ import annotations

import argparse
from pathlib import Path

from finops_r.integrated_qualification_stage5 import verify_integrated_qualification
from finops_r.json_safety import strict_json_dumps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify a FinOps-R Stage 5 qualification artifact"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/qualification/integrated_stage5_v1r1"),
    )
    args = parser.parse_args()
    result = verify_integrated_qualification(args.artifact_dir)
    print(strict_json_dumps(result, indent=2, sort_keys=True))
    return 0 if result["integrated_qualification_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
