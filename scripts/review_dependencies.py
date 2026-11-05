"""Fail when a dependency scan finds a vulnerability absent from the base scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def findings(path: Path, expected_inputs: int) -> set[tuple[str, str, str, str]]:
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    if len(results) != expected_inputs or any(not entry["packages"] for entry in results):
        raise ValueError(f"Incomplete dependency scan: {path.name}")

    return {
        (package["ecosystem"], package["name"], package["version"], advisory["id"])
        for result in results
        for item in result["packages"]
        for package in (item["package"],)
        for advisory in item.get("vulnerabilities", ())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("--inputs", type=int, default=1)
    args = parser.parse_args()

    base = findings(args.base, args.inputs)
    head = findings(args.head, args.inputs)
    introduced = head - base
    print(f"Dependency advisories: {len(base)} before, {len(head)} after.")
    for ecosystem, name, version, advisory in sorted(introduced):
        print(f"{ecosystem}: {name}=={version} introduces {advisory}")
    return int(bool(introduced))


if __name__ == "__main__":
    raise SystemExit(main())
