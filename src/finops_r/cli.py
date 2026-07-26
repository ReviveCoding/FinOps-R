from __future__ import annotations

import argparse
import sys
from pathlib import Path, PureWindowsPath

from . import __version__
from .artifacts import atomic_json, verify_artifact_manifest
from .datasets import inspect_zip, load_dataset_registry, stage_iso20022
from .json_safety import require_json_object, strict_json_dumps, strict_json_load
from .path_safety import is_portable_component
from .pipeline import (
    assess_request,
    generate_only,
    load_production_bundle,
    regenerate_report,
    run_project,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finops-r", description="FinOps-R controlled benchmark CLI"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="re-raise runtime errors with their original traceback",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="execute the full data-to-release workflow")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-root", type=Path)

    generate = subparsers.add_parser("generate", help="generate and validate TradeOps-Sim data")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, default=Path("data/generated/tradeops_sim.csv"))

    report = subparsers.add_parser("report", help="regenerate canonical figures and reports")
    report.add_argument("--run-dir", type=Path, required=True)

    inspect = subparsers.add_parser("inspect", help="print a concise release summary")
    inspect.add_argument("--run-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify every artifact before loading a model")
    verify.add_argument("--run-dir", type=Path, required=True)

    assess = subparsers.add_parser(
        "assess", help="assess one synthetic case through the production bundle"
    )
    assess.add_argument("--run-dir", type=Path, required=True)
    assess.add_argument("--request", type=Path, required=True)

    serve = subparsers.add_parser("serve", help="serve the verified local assessment API")
    serve.add_argument("--run-dir", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    rollback = subparsers.add_parser(
        "rollback", help="set a prior passing run as the local champion"
    )
    rollback.add_argument("--run-id", required=True)
    rollback.add_argument("--project-root", type=Path, default=Path.cwd())

    data = subparsers.add_parser("data", help="inspect and stage external dataset snapshots")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)

    data_list = data_subparsers.add_parser("list", help="print the governed dataset registry")
    data_list.add_argument(
        "--registry", type=Path, default=Path(__file__).with_name("dataset_registry.json")
    )

    data_verify = data_subparsers.add_parser(
        "verify-zip", help="fail-closed ZIP integrity and extraction-safety audit"
    )
    data_verify.add_argument("--archive", type=Path, required=True)

    data_stage_iso = data_subparsers.add_parser(
        "stage-iso", help="validate and atomically stage the ISO 20022-like synthetic snapshot"
    )
    data_stage_iso.add_argument("--archive", type=Path, required=True)
    data_stage_iso.add_argument(
        "--output", type=Path, default=Path("data/external/iso20022_synthetic_v1")
    )
    return parser


def _safe_run_id(value: str) -> str:
    path = Path(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or value != value.strip()
        or value.startswith(".")
        or "/" in value
        or "\\" in value
        or path.name != value
        or windows.drive
        or len(value) > 128
        or not is_portable_component(value)
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in value
        )
    ):
        raise ValueError("run-id must be one safe filename component")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            run_dir = run_project(args.config, args.output_root)
            release = require_json_object(
                strict_json_load(run_dir / "release_decision.json"), source="release decision"
            )
            print(
                strict_json_dumps(
                    {"run_dir": str(run_dir), "release": release}, indent=2, sort_keys=True
                )
            )
        elif args.command == "generate":
            path = generate_only(args.config, args.output)
            print(strict_json_dumps({"generated": str(path)}, indent=2))
        elif args.command == "report":
            output_run_dir, figures = regenerate_report(args.run_dir)
            print(
                strict_json_dumps(
                    {
                        "source_run_dir": str(args.run_dir),
                        "output_run_dir": str(output_run_dir),
                        "figures": figures,
                    },
                    indent=2,
                )
            )
        elif args.command == "inspect":
            run_dir = args.run_dir
            integrity = verify_artifact_manifest(run_dir)
            release = require_json_object(
                strict_json_load(run_dir / "release_decision.json"), source="release decision"
            )
            print(
                strict_json_dumps(
                    {"release": release, "integrity": integrity}, indent=2, sort_keys=True
                )
            )
        elif args.command == "verify":
            result = verify_artifact_manifest(args.run_dir)
            print(strict_json_dumps(result, indent=2, sort_keys=True))
        elif args.command == "assess":
            result = assess_request(args.run_dir, args.request.resolve())
            print(strict_json_dumps(result, indent=2, sort_keys=True))
        elif args.command == "serve":
            try:
                import uvicorn
            except ImportError as exc:
                raise RuntimeError("Install the API extra with: pip install -e '.[api]'") from exc
            from .api import create_app

            uvicorn.run(create_app(args.run_dir), host=args.host, port=args.port)
        elif args.command == "rollback":
            root = args.project_root.resolve()
            run_id = _safe_run_id(args.run_id)
            run_dir = root / "artifacts" / "runs" / run_id
            release_path = run_dir / "release_decision.json"
            bundle_path = run_dir / "models" / "production_bundle.joblib"
            if not release_path.exists() or not bundle_path.exists():
                raise FileNotFoundError(
                    "rollback target is missing release decision or production bundle"
                )
            load_production_bundle(run_dir)
            release = require_json_object(strict_json_load(release_path), source="release decision")
            if release["status"] != "PROMOTE":
                raise ValueError("rollback target was not a promoted run")
            pointer = root / "artifacts" / "releases" / "champion.json"
            atomic_json(
                pointer,
                {
                    "run_id": run_id,
                    "run_dir": f"artifacts/runs/{run_id}",
                    "candidate": release["production_champion"],
                },
            )
            print(strict_json_dumps({"rollback": "PASS", "champion": run_id}, indent=2))
        elif args.command == "data":
            if args.data_command == "list":
                print(
                    strict_json_dumps(
                        load_dataset_registry(args.registry), indent=2, sort_keys=True
                    )
                )
            elif args.data_command == "verify-zip":
                print(strict_json_dumps(inspect_zip(args.archive), indent=2, sort_keys=True))
            elif args.data_command == "stage-iso":
                print(
                    strict_json_dumps(
                        stage_iso20022(args.archive, args.output), indent=2, sort_keys=True
                    )
                )
        return 0
    except KeyboardInterrupt:
        if args.debug:
            raise
        print(
            strict_json_dumps(
                {
                    "status": "ERROR",
                    "type": "KeyboardInterrupt",
                    "message": "operation interrupted by user",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        if args.debug:
            raise
        print(
            strict_json_dumps(
                {"status": "ERROR", "type": type(exc).__name__, "message": str(exc)}, indent=2
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
