from __future__ import annotations

import argparse
from pathlib import Path

from finops_r.e4_evidence import verify_hosted_e4_evidence
from finops_r.json_safety import strict_json_dumps


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a complete hosted E4 evidence set")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_hosted_e4_evidence(
        args.evidence_dir,
        expected_commit=args.expected_commit,
    )
    rendered = strict_json_dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(rendered.encode("utf-8"))
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
