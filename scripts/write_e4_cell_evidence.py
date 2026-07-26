from __future__ import annotations

import argparse
import os
from pathlib import Path

from finops_r.e4_evidence import write_hosted_cell_evidence
from finops_r.json_safety import strict_json_dumps


def main() -> int:
    parser = argparse.ArgumentParser(description="Write one GitHub-hosted E4 matrix record")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage4-dir",
        type=Path,
        default=Path("artifacts/qualification/orchestration_stage4_v1r2"),
    )
    parser.add_argument(
        "--stage5-dir",
        type=Path,
        default=Path("artifacts/qualification/integrated_stage5_v1r1"),
    )
    args = parser.parse_args()
    result = write_hosted_cell_evidence(
        output_path=args.output,
        environment=os.environ,
        stage4_dir=args.stage4_dir,
        stage5_dir=args.stage5_dir,
    )
    print(strict_json_dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
