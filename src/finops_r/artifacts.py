from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
import time
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import pandas as pd

from .json_safety import StrictJSONError, strict_json_dump, strict_json_dumps, strict_json_load
from .path_safety import is_portable_component


class ArtifactVerificationError(RuntimeError):
    """Raised when a local release artifact fails its integrity contract."""


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_METADATA = {"artifact_manifest.json", "artifact_integrity.json"}
_IS_WINDOWS = os.name == "nt"
_WINDOWS_REPLACE_RETRY_SECONDS = 5.0
_WINDOWS_REPLACE_MAX_DELAY_SECONDS = 0.05


def _flush_and_sync(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _atomic_replace(source: Path, destination: Path) -> None:
    """Replace destination, tolerating transient Windows sharing violations."""
    deadline = time.monotonic() + _WINDOWS_REPLACE_RETRY_SECONDS
    delay = 0.001
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if not _IS_WINDOWS or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, _WINDOWS_REPLACE_MAX_DELAY_SECONDS)


def _is_manifest_metadata(relative: Path) -> bool:
    return relative.as_posix() in _MANIFEST_METADATA


def _contains_hidden_part(relative: Path) -> bool:
    return any(part.startswith(".") for part in relative.parts)


def _portable_relative_key(relative: Path) -> str:
    return unicodedata.normalize("NFC", relative.as_posix()).casefold()


def _verified_tree_inventory(root: Path) -> list[Path]:
    items = list(root.rglob("*"))
    portable_seen: dict[str, Path] = {}
    for item in items:
        if item.is_symlink():
            raise ArtifactVerificationError(f"artifact tree contains a symlink: {item}")
        relative = item.relative_to(root)
        item_stat = item.stat(follow_symlinks=False)
        mode = item_stat.st_mode
        if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
            raise ArtifactVerificationError(
                f"artifact tree contains a special filesystem object: {relative}"
            )
        if stat.S_ISREG(mode) and item_stat.st_nlink != 1:
            raise ArtifactVerificationError(f"artifact tree contains a hardlink: {relative}")
        if _contains_hidden_part(relative):
            raise ArtifactVerificationError(
                f"artifact tree contains hidden or residual temporary content: {relative}"
            )
        if any(not is_portable_component(part) for part in relative.parts):
            raise ArtifactVerificationError(
                f"artifact tree contains a non-portable path: {relative}"
            )
        key = _portable_relative_key(relative)
        if key in portable_seen:
            raise ArtifactVerificationError(
                "artifact tree contains a cross-platform path collision: "
                f"{portable_seen[key]} and {relative}"
            )
        portable_seen[key] = relative
    return items


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    payload = strict_json_dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            strict_json_dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            _flush_and_sync(handle)
        _atomic_replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            _flush_and_sync(handle)
        _atomic_replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
            newline="",
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
            _flush_and_sync(handle)
        _atomic_replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_joblib(path: Path, value: Any, staging_root: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = staging_root or path.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    owner = path.parent.parent.name.lstrip(".") if staging_root is not None else path.stem
    with tempfile.TemporaryDirectory(
        dir=staging_parent, prefix=f".{owner}.{path.name}.staging."
    ) as directory:
        temporary = Path(directory) / path.name
        joblib.dump(value, temporary, compress=("zlib", 3))
        with temporary.open("r+b") as handle:
            _flush_and_sync(handle)
        _atomic_replace(temporary, path)


def cleanup_empty_joblib_staging(root: Path, owner: str | None = None) -> list[str]:
    """Remove delayed empty joblib staging directories, never their contents."""
    removed: list[str] = []
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.name.startswith(".")
        and ".joblib.staging." in path.name
        and (owner is None or path.name.startswith(f".{owner}."))
    )
    for path in candidates:
        relative = path.name
        if path.is_symlink() or not path.is_dir():
            raise ArtifactVerificationError(f"unsafe joblib staging residue: {relative}")
        if any(path.iterdir()):
            raise ArtifactVerificationError(f"non-empty joblib staging residue: {relative}")
        path.rmdir()
        removed.append(relative)
    return removed


def cleanup_empty_directory_tree(path: Path) -> list[str]:
    """Remove one exact empty directory tree and refuse any file or symlink."""
    if not path.exists() and not path.is_symlink():
        return []
    if path.is_symlink() or not path.is_dir():
        raise ArtifactVerificationError(f"unsafe publication staging residue: {path}")
    descendants = sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for item in descendants:
        if item.is_symlink() or not item.is_dir():
            raise ArtifactVerificationError(
                f"non-empty publication staging residue: {item.relative_to(path)}"
            )
    removed = [item.relative_to(path).as_posix() for item in descendants]
    for item in descendants:
        item.rmdir()
    path.rmdir()
    return removed


def publish_verified_tree(staging: Path, destination: Path) -> None:
    """Copy, verify, and atomically expose a tree under its final public name."""
    if staging.is_symlink() or not staging.is_dir():
        raise ArtifactVerificationError("publication staging must be a real directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    publication = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.publishing.",
        )
    )
    try:
        shutil.copytree(staging, publication, dirs_exist_ok=True)
        verify_artifact_manifest(publication)
        # Remove the private source before exposing the verified copy.  If cleanup
        # fails, callers see a failure and no public destination—not a release
        # which became visible immediately before an exception was raised.
        shutil.rmtree(staging)
        os.replace(publication, destination)
    except BaseException:
        if publication.is_symlink():
            raise ArtifactVerificationError("publication temporary unexpectedly became a symlink")
        if publication.exists():
            shutil.rmtree(publication)
        raise


def artifact_manifest(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactVerificationError("artifact root must be a real directory")
    records: list[dict[str, Any]] = []
    inventory = _verified_tree_inventory(root)
    for path in sorted(item for item in inventory if item.is_file()):
        relative = path.relative_to(root)
        if _is_manifest_metadata(relative):
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def verify_artifact_manifest(root: Path) -> dict[str, Any]:
    if root.is_symlink():
        raise ArtifactVerificationError("artifact root must not be a symlink")
    root = root.resolve()
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ArtifactVerificationError("artifact manifest is missing or is a symlink")
    try:
        records = strict_json_load(manifest_path)
    except (OSError, StrictJSONError) as exc:
        raise ArtifactVerificationError("artifact manifest is unreadable") from exc
    if not isinstance(records, list) or not records:
        raise ArtifactVerificationError("artifact manifest must be a non-empty list")
    checked = 0
    seen: set[str] = set()
    portable_seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ArtifactVerificationError("artifact manifest record has an invalid schema")
        raw_path = record["path"]
        byte_count = record["bytes"]
        checksum = record["sha256"]
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
            raise ArtifactVerificationError("artifact manifest contains an invalid path")
        relative_posix = PurePosixPath(raw_path)
        normalized = relative_posix.as_posix()
        portable_key = unicodedata.normalize("NFC", normalized).casefold()
        if normalized in seen:
            raise ArtifactVerificationError("artifact manifest contains a duplicate path")
        if portable_key in portable_seen:
            raise ArtifactVerificationError(
                "artifact manifest contains a cross-platform path collision"
            )
        if (
            relative_posix.is_absolute()
            or normalized != raw_path
            or ".." in relative_posix.parts
            or "." in relative_posix.parts
            or _contains_hidden_part(Path(*relative_posix.parts))
            or any(not is_portable_component(part) for part in relative_posix.parts)
            or normalized in _MANIFEST_METADATA
        ):
            raise ArtifactVerificationError(
                "artifact manifest contains an unsafe or duplicate path"
            )
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ArtifactVerificationError("artifact byte count must be a non-negative integer")
        if not isinstance(checksum, str) or _SHA256_PATTERN.fullmatch(checksum) is None:
            raise ArtifactVerificationError("artifact checksum must be lowercase SHA-256")
        seen.add(normalized)
        portable_seen.add(portable_key)
        relative = Path(*relative_posix.parts)
        candidate = root / relative
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ArtifactVerificationError(f"artifact path contains a symlink: {relative}")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ArtifactVerificationError(f"artifact is missing: {relative}") from exc
        if root not in resolved.parents or not candidate.is_file():
            raise ArtifactVerificationError(f"artifact is missing or unsafe: {relative}")
        if candidate.stat().st_size != byte_count:
            raise ArtifactVerificationError(f"artifact size mismatch: {relative}")
        if sha256_file(candidate) != checksum:
            raise ArtifactVerificationError(f"artifact checksum mismatch: {relative}")
        checked += 1
    actual: set[str] = set()
    for item in _verified_tree_inventory(root):
        relative = item.relative_to(root)
        if not item.is_file():
            continue
        if _is_manifest_metadata(relative):
            continue
        actual.add(relative.as_posix())
    if seen != actual:
        missing = sorted(seen - actual)
        untracked = sorted(actual - seen)
        raise ArtifactVerificationError(
            f"artifact inventory mismatch: missing={missing}, untracked={untracked}"
        )
    return {"status": "PASS", "checked": checked}
