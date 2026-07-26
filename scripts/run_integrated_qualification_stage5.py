from __future__ import annotations

import argparse
from pathlib import Path

from finops_r.integrated_qualification_stage5 import run_integrated_qualification
from finops_r.json_safety import strict_json_dumps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the locked FinOps-R integrated Stage 5 qualification"
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("data/policy_cards/integrated_qualification_stage5_locked.json"),
    )
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage4-artifact-dir",
        type=Path,
        default=Path("artifacts/qualification/orchestration_stage4_v1r2"),
    )
    parser.add_argument(
        "--upstream-identity",
        type=Path,
        default=Path("UPSTREAM_SOURCE_SNAPSHOT.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/qualification/integrated_stage5_v1r1"),
    )
    args = parser.parse_args()
    result = run_integrated_qualification(
        contract_path=args.contract,
        model_run_dir=args.model_run_dir,
        stage4_artifact_dir=args.stage4_artifact_dir,
        upstream_identity_path=args.upstream_identity,
        output_dir=args.output_dir,
    )
    print(strict_json_dumps(result, indent=2, sort_keys=True))
    return 0 if result["integrated_qualification_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
