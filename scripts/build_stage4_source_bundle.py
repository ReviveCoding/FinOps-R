from __future__ import annotations

import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from finops_r.artifacts import atomic_json, sha256_file
from finops_r.json_safety import strict_json_dumps
from finops_r.stage4_qualification import verify_stage4_qualification

BUNDLE_ROOT = "JPMC01_FinOps-R_v0.4.0rc1"
FINAL_QUALIFICATION = PurePosixPath("artifacts/qualification/orchestration_stage4_v1r2")
EXCLUDED_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "finops_r.egg-info",
}


def _included(relative: PurePosixPath) -> bool:
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.suffix in {".pyc", ".pyo"}:
        return False
    if relative.parts and relative.parts[0] == "artifacts":
        return relative.is_relative_to(FINAL_QUALIFICATION)
    return True


def _inventory(root: Path) -> list[tuple[PurePosixPath, Path, int, str]]:
    records: list[tuple[PurePosixPath, Path, int, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source bundle cannot contain a symlink: {path}")
        if not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if not _included(relative):
            continue
        records.append((relative, path, path.stat().st_size, sha256_file(path)))
    if not records:
        raise ValueError("source bundle inventory is empty")
    return records


def _entry(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def build_bundle(
    *,
    source_root: Path,
    output_zip: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    records = _inventory(source_root)
    manifest_text = "".join(
        f"{checksum}  {size}  {relative.as_posix()}\n" for relative, _, size, checksum in records
    )
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_zip.parent,
            prefix=f".{output_zip.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for relative, path, _, _ in records:
                archive.writestr(
                    _entry(f"{BUNDLE_ROOT}/{relative.as_posix()}"),
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
            archive.writestr(
                _entry(f"{BUNDLE_ROOT}/STAGE4_SOURCE_BUNDLE_SHA256.txt"),
                manifest_text.encode("utf-8"),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
        with zipfile.ZipFile(temporary, mode="r") as archive:
            archive.testzip()
            if len(archive.infolist()) != len(records) + 1:
                raise ValueError("source bundle ZIP entry count is invalid")
            for relative, _, size, checksum in records:
                value = archive.read(f"{BUNDLE_ROOT}/{relative.as_posix()}")
                if len(value) != size or hashlib.sha256(value).hexdigest() != checksum:
                    raise ValueError(f"source bundle entry mismatch: {relative}")
            if archive.read(
                f"{BUNDLE_ROOT}/STAGE4_SOURCE_BUNDLE_SHA256.txt"
            ) != manifest_text.encode("utf-8"):
                raise ValueError("source bundle manifest bytes changed")
        temporary.replace(output_zip)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    qualification = verify_stage4_qualification(source_root / FINAL_QUALIFICATION.as_posix())
    result = {
        "schema_version": "1.0",
        "bundle_file": output_zip.name,
        "bundle_bytes": output_zip.stat().st_size,
        "bundle_sha256": sha256_file(output_zip),
        "source_file_count": len(records),
        "zip_entry_count": len(records) + 1,
        "qualification": qualification,
        "upstream_snapshot_sha256": (
            "c6c212540e2b37acd8cdbc809a9b5c3a41948aadb30416f47b5f8f7ad30ceefa"
        ),
        "status": "PASS",
    }
    atomic_json(output_manifest, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and verify the deterministic FinOps-R Stage 4 source bundle"
    )
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("JPMC01_FinOps-R_v0.4.0rc1_stage4_source.zip"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("JPMC01_FinOps-R_v0.4.0rc1_stage4_source_manifest.json"),
    )
    args = parser.parse_args()
    result = build_bundle(
        source_root=args.source_root,
        output_zip=args.output,
        output_manifest=args.manifest,
    )
    print(strict_json_dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
