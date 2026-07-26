from __future__ import annotations

import hashlib
import math
import os
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd

from .json_safety import StrictJSONError, strict_json_dumps, strict_json_loads
from .path_safety import is_portable_component


ISO_REQUIRED_MEMBERS = {
    "iso20022_analytics.csv",
    "pacs002.csv",
    "pacs008.csv",
}
ISO_MAX_SOURCE_ARCHIVE_BYTES = 300_000_000

ISO_ANALYTICS_COLUMNS = {
    "MessageId",
    "EndToEndId",
    "TxDateTime",
    "StatusTimestamp",
    "ProcessingTimeSecs",
    "IntrBkSttlmDt",
    "InstdAmt",
    "Currency",
    "DbtrAgtBIC",
    "CdtrAgtBIC",
    "DbtrCountry",
    "CdtrCountry",
    "PurposeCode",
    "TxSts",
    "RejectionReason",
}


class DataContractError(ValueError):
    """Raised when an external dataset violates a declared, fail-closed contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _copy_bounded_archive(source: Path, destination: Path, max_bytes: int) -> int:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("archive copy limit must be a positive integer")
    copied = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            copied += len(block)
            if copied > max_bytes:
                raise DataContractError(f"source archive exceeds {max_bytes} compressed bytes")
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    return copied


def _safe_zip_name(name: str) -> bool:
    if not name or "\\" in name or "\x00" in name:
        return False
    windows = PureWindowsPath(name)
    if windows.is_absolute() or windows.drive:
        return False
    path = PurePosixPath(name)
    if path.is_absolute():
        return False
    components = name[:-1].split("/") if name.endswith("/") else name.split("/")
    if not components or any(component in {"", ".", ".."} for component in components):
        return False
    return all(is_portable_component(component) for component in components)


def _portable_zip_key(name: str) -> str:
    """Return the case-insensitive, NFC-normalized extraction namespace key."""
    return unicodedata.normalize("NFC", name.rstrip("/")).casefold()


def inspect_zip(
    archive: Path,
    *,
    required_members: set[str] | None = None,
    max_members: int = 50_000,
    max_uncompressed_bytes: int = 2_000_000_000,
    max_compression_ratio: float = 500.0,
) -> dict[str, Any]:
    """Read and CRC-check a ZIP without extracting it.

    The checks reject traversal, duplicate names, encrypted members, symlinks,
    unbounded expansion and suspicious compression ratios before any file is staged.
    """
    if isinstance(max_members, bool) or not isinstance(max_members, int) or max_members < 1:
        raise ValueError("member limit must be a positive integer")
    if (
        isinstance(max_uncompressed_bytes, bool)
        or not isinstance(max_uncompressed_bytes, int)
        or max_uncompressed_bytes < 1
    ):
        raise ValueError("uncompressed byte limit must be a positive integer")
    if (
        isinstance(max_compression_ratio, bool)
        or not isinstance(max_compression_ratio, (int, float))
        or not math.isfinite(float(max_compression_ratio))
        or max_compression_ratio <= 0.0
    ):
        raise ValueError("compression ratio limit must be a finite positive number")
    archive = archive.resolve()
    if not archive.is_file():
        raise DataContractError(f"not a readable ZIP archive: {archive}")
    try:
        archive_stream = archive.open("rb")
    except OSError as exc:
        raise DataContractError(f"not a readable ZIP archive: {archive}") from exc
    if not zipfile.is_zipfile(archive_stream):
        archive_stream.close()
        raise DataContractError(f"not a readable ZIP archive: {archive}")
    archive_stream.seek(0)

    with archive_stream, zipfile.ZipFile(archive_stream) as bundle:
        members = bundle.infolist()
        if not members:
            raise DataContractError("archive is empty")
        if len(members) > max_members:
            raise DataContractError(f"archive has {len(members)} members; limit is {max_members}")

        names = [member.filename for member in members]
        duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicate_names:
            raise DataContractError(f"archive contains duplicate member names: {duplicate_names}")

        unsafe_names = sorted(name for name in names if not _safe_zip_name(name))
        if unsafe_names:
            raise DataContractError(f"archive contains unsafe member names: {unsafe_names}")

        portable_keys = [_portable_zip_key(name) for name in names]
        portable_collisions = sorted(
            key for key, count in Counter(portable_keys).items() if count > 1
        )
        if portable_collisions:
            raise DataContractError(
                f"archive contains cross-platform name collisions: {portable_collisions}"
            )

        file_keys = {
            _portable_zip_key(member.filename) for member in members if not member.is_dir()
        }
        parent_collisions: list[str] = []
        for key in portable_keys:
            components = key.split("/")
            for depth in range(1, len(components)):
                parent = "/".join(components[:depth])
                if parent in file_keys:
                    parent_collisions.append(f"{parent} -> {key}")
        if parent_collisions:
            raise DataContractError(
                f"archive contains file/directory name collisions: {sorted(set(parent_collisions))}"
            )

        encrypted = sorted(member.filename for member in members if member.flag_bits & 0x1)
        if encrypted:
            raise DataContractError(f"encrypted members are not supported: {encrypted}")

        type_marker_mismatches = sorted(
            member.filename
            for member in members
            if (file_type := stat.S_IFMT((member.external_attr >> 16) & 0xFFFF))
            in {stat.S_IFREG, stat.S_IFDIR}
            and member.is_dir() != (file_type == stat.S_IFDIR)
        )
        if type_marker_mismatches:
            raise DataContractError(
                "archive member type marker disagrees with its external mode: "
                f"{type_marker_mismatches}"
            )

        symlinks = sorted(
            member.filename
            for member in members
            if stat.S_IFMT((member.external_attr >> 16) & 0xFFFF) == stat.S_IFLNK
        )
        if symlinks:
            raise DataContractError(f"archive contains symbolic links: {symlinks}")

        special_members = sorted(
            member.filename
            for member in members
            if stat.S_IFMT((member.external_attr >> 16) & 0xFFFF)
            not in {0, stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK}
        )
        if special_members:
            raise DataContractError(
                f"archive contains special filesystem members: {special_members}"
            )

        total_uncompressed = sum(member.file_size for member in members)
        total_compressed = sum(member.compress_size for member in members)
        if total_uncompressed > max_uncompressed_bytes:
            raise DataContractError(
                f"archive expands to {total_uncompressed} bytes; limit is {max_uncompressed_bytes}"
            )
        ratios = [
            member.file_size / max(1, member.compress_size)
            for member in members
            if not member.is_dir() and member.file_size
        ]
        maximum_ratio = max(ratios, default=0.0)
        if maximum_ratio > max_compression_ratio:
            raise DataContractError(
                f"archive compression ratio {maximum_ratio:.1f} exceeds {max_compression_ratio:.1f}"
            )

        if required_members is not None:
            missing = sorted(required_members - set(names))
            if missing:
                raise DataContractError(f"archive is missing required members: {missing}")

        corrupt_member = bundle.testzip()
        if corrupt_member is not None:
            raise DataContractError(f"CRC check failed for archive member: {corrupt_member}")
        archive_sha256 = _sha256_stream(archive_stream)

    return {
        "status": "PASS",
        "archive": archive.name,
        "sha256": archive_sha256,
        "member_count": len(members),
        "member_sample": sorted(names)[:50],
        "members_truncated": len(names) > 50,
        "compressed_bytes": total_compressed,
        "uncompressed_bytes": total_uncompressed,
        "maximum_compression_ratio": maximum_ratio,
    }


def _read_iso_frames(archive: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(archive) as bundle:
        for member in sorted(ISO_REQUIRED_MEMBERS):
            with bundle.open(member) as stream:
                frames[member] = pd.read_csv(stream)
    return frames


def _require_columns(frame: pd.DataFrame, expected: set[str], name: str) -> None:
    missing = sorted(expected - set(frame.columns))
    unexpected = sorted(set(frame.columns) - expected)
    if missing or unexpected:
        raise DataContractError(
            f"{name} schema mismatch; missing={missing}, unexpected={unexpected}"
        )


def validate_iso20022_frames(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Validate the supplied flattened ISO 20022-like synthetic snapshot."""
    if set(frames) != ISO_REQUIRED_MEMBERS:
        raise DataContractError("ISO frame set must contain exactly the three declared CSV files")

    analytics = frames["iso20022_analytics.csv"].copy()
    pacs002 = frames["pacs002.csv"].copy()
    pacs008 = frames["pacs008.csv"].copy()
    _require_columns(analytics, ISO_ANALYTICS_COLUMNS, "iso20022_analytics.csv")
    _require_columns(
        pacs002,
        {"OriginalMessageId", "TxSts", "RejectionReason", "StatusTimestamp"},
        "pacs002.csv",
    )
    _require_columns(
        pacs008,
        {
            "MessageId",
            "EndToEndId",
            "TxDateTime",
            "IntrBkSttlmDt",
            "InstdAmt",
            "Currency",
            "DbtrAgtBIC",
            "CdtrAgtBIC",
            "DbtrCountry",
            "CdtrCountry",
            "PurposeCode",
        },
        "pacs008.csv",
    )

    row_counts = {name: len(frame) for name, frame in frames.items()}
    if row_counts["iso20022_analytics.csv"] < 10 or len(set(row_counts.values())) != 1:
        raise DataContractError(f"paired files must be non-empty and row-aligned: {row_counts}")

    for frame, column, name in (
        (analytics, "MessageId", "analytics"),
        (pacs002, "OriginalMessageId", "pacs002"),
        (pacs008, "MessageId", "pacs008"),
    ):
        if frame[column].isna().any() or frame[column].duplicated().any():
            raise DataContractError(f"{name} message identifiers must be non-null and unique")

    identity_columns = (
        "MessageId",
        "EndToEndId",
        "Currency",
        "DbtrAgtBIC",
        "CdtrAgtBIC",
        "DbtrCountry",
        "CdtrCountry",
        "PurposeCode",
    )
    for column in identity_columns:
        values = analytics[column]
        invalid = values.isna() | values.map(
            lambda value: not isinstance(value, str) or not value.strip() or value != value.strip()
        )
        if invalid.any():
            raise DataContractError(f"analytics {column} values must be non-empty trimmed strings")
    if analytics["EndToEndId"].duplicated().any():
        raise DataContractError("analytics end-to-end identifiers must be unique")

    ids = set(analytics["MessageId"].astype(str))
    if ids != set(pacs002["OriginalMessageId"].astype(str)) or ids != set(
        pacs008["MessageId"].astype(str)
    ):
        raise DataContractError("pacs.008, pacs.002 and analytics message identifiers do not match")

    statuses = set(analytics["TxSts"].dropna().astype(str))
    if analytics["TxSts"].isna().any() or not statuses or not statuses.issubset({"ACSP", "RJCT"}):
        raise DataContractError(f"unsupported transaction statuses: {sorted(statuses)}")

    joined_status = analytics[["MessageId", "TxSts", "RejectionReason", "StatusTimestamp"]].merge(
        pacs002,
        left_on="MessageId",
        right_on="OriginalMessageId",
        how="outer",
        validate="one_to_one",
        suffixes=("_analytics", "_pacs002"),
    )
    for column in ("TxSts", "StatusTimestamp"):
        if (
            not joined_status[f"{column}_analytics"]
            .astype(str)
            .equals(joined_status[f"{column}_pacs002"].astype(str))
        ):
            raise DataContractError(f"analytics and pacs.002 disagree on {column}")
    if (
        not joined_status["RejectionReason_analytics"]
        .fillna("")
        .astype(str)
        .equals(joined_status["RejectionReason_pacs002"].fillna("").astype(str))
    ):
        raise DataContractError("analytics and pacs.002 disagree on RejectionReason")

    analytics_messages = analytics.set_index("MessageId").sort_index()
    pacs008_messages = pacs008.set_index("MessageId").sort_index()
    for column in sorted(set(pacs008.columns) - {"MessageId"}):
        if (
            not analytics_messages[column]
            .fillna("")
            .astype(str)
            .equals(pacs008_messages[column].fillna("").astype(str))
        ):
            raise DataContractError(f"analytics and pacs.008 disagree on {column}")

    rejected = analytics["TxSts"] == "RJCT"
    rejection_reasons = analytics.loc[rejected, "RejectionReason"]
    invalid_rejection_reasons = rejection_reasons.isna() | rejection_reasons.map(
        lambda value: not isinstance(value, str) or not value.strip() or value != value.strip()
    )
    if invalid_rejection_reasons.any():
        raise DataContractError(
            "rejected transactions require a non-empty trimmed rejection reason"
        )
    if analytics.loc[~rejected, "RejectionReason"].notna().any():
        raise DataContractError("accepted transactions must not carry a rejection reason")

    transaction_time = pd.to_datetime(analytics["TxDateTime"], errors="coerce", utc=True)
    status_time = pd.to_datetime(analytics["StatusTimestamp"], errors="coerce", utc=True)
    if transaction_time.isna().any() or status_time.isna().any():
        raise DataContractError("transaction and status timestamps must be parseable")
    settlement_date = pd.to_datetime(
        analytics["IntrBkSttlmDt"], errors="coerce", utc=True, format="%Y-%m-%d"
    )
    if settlement_date.isna().any() or (settlement_date.dt.date < transaction_time.dt.date).any():
        raise DataContractError(
            "interbank settlement date must be parseable and not precede the transaction date"
        )
    observed_seconds = (status_time - transaction_time).dt.total_seconds()
    declared_seconds = pd.to_numeric(analytics["ProcessingTimeSecs"], errors="coerce")
    if declared_seconds.isna().any() or (declared_seconds < 0).any():
        raise DataContractError("processing time must be finite and non-negative")
    if not (observed_seconds == declared_seconds).all():
        raise DataContractError("processing time does not match the timestamp difference")

    amount = pd.to_numeric(analytics["InstdAmt"], errors="coerce")
    if (
        amount.isna().any()
        or not np.isfinite(amount.to_numpy(dtype=float)).all()
        or (amount <= 0).any()
    ):
        raise DataContractError("instructed amounts must be finite and positive")

    dates = transaction_time.dt.date.astype(str)
    return {
        "status": "PASS",
        "row_counts": row_counts,
        "date_range": {"start": str(dates.min()), "end": str(dates.max())},
        "status_counts": {
            str(key): int(value) for key, value in analytics["TxSts"].value_counts().items()
        },
        "rejection_reason_counts": {
            str(key): int(value)
            for key, value in analytics.loc[rejected, "RejectionReason"].value_counts().items()
        },
        "currency_counts": {
            str(key): int(value) for key, value in analytics["Currency"].value_counts().items()
        },
    }


def _temporal_assignments(analytics: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(analytics["TxDateTime"], utc=True)
    order = timestamps.sort_values(kind="stable").index
    count = len(order)
    train_end = max(1, int(count * 0.60))
    validation_end = max(train_end + 1, int(count * 0.80))
    split = pd.Series("test", index=analytics.index, dtype="string")
    split.loc[order[:train_end]] = "train"
    split.loc[order[train_end:validation_end]] = "validation"
    return pd.DataFrame(
        {
            "MessageId": analytics["MessageId"].astype(str),
            "TxDateTime": analytics["TxDateTime"].astype(str),
            "split": split,
        }
    )


def stage_iso20022(archive: Path, output: Path) -> dict[str, Any]:
    """Validate and atomically stage the user-provided ISO 20022-like snapshot."""
    archive = archive.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        # Work only from one private snapshot.  The source path may be replaced by
        # another process after inspection; reopening it would disconnect the
        # recorded hash and validation result from the bytes ultimately staged.
        snapshot = staging / ".source-archive.zip"
        _copy_bounded_archive(archive, snapshot, ISO_MAX_SOURCE_ARCHIVE_BYTES)
        archive_report = inspect_zip(
            snapshot,
            required_members=ISO_REQUIRED_MEMBERS,
            max_members=10,
            max_uncompressed_bytes=250_000_000,
            max_compression_ratio=100.0,
        )
        frames = _read_iso_frames(snapshot)
        validation = validate_iso20022_frames(frames)
        assignments = _temporal_assignments(frames["iso20022_analytics.csv"])
        split_counts = {
            str(key): int(value) for key, value in assignments["split"].value_counts().items()
        }
        manifest = {
            "schema_version": "1.0",
            "dataset_id": "iso20022_payments_analytics_synthetic_v1",
            "classification": "user-provided project-generated synthetic ISO 20022-like analytics",
            "official_iso_dataset": False,
            "production_bank_data": False,
            "source_archive": archive.name,
            "source_sha256": archive_report["sha256"],
            "archive": {**archive_report, "archive": archive.name},
            "validation": validation,
            "temporal_split_counts": split_counts,
            "claim_boundary": (
                "Suitable for data-contract and workflow demonstrations only; not evidence of "
                "production payment performance or ISO 20022 conformance."
            ),
        }
        with zipfile.ZipFile(snapshot) as bundle:
            for member in sorted(ISO_REQUIRED_MEMBERS):
                (staging / member).write_bytes(bundle.read(member))
        assignments.to_csv(staging / "temporal_splits.csv", index=False)
        (staging / "dataset_manifest.json").write_text(
            strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "README.md").write_text(
            "# ISO 20022-like synthetic snapshot\n\n"
            "This directory was generated by `finops-r data stage-iso`. The source is a "
            "user-provided, project-generated synthetic flattened CSV snapshot. It is not an "
            "official ISO dataset and contains no real bank transactions. See "
            "`dataset_manifest.json` for hashes, validation evidence and the claim boundary.\n",
            encoding="utf-8",
        )
        snapshot.unlink()
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def load_dataset_registry(path: Path) -> dict[str, Any]:
    try:
        registry = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, StrictJSONError) as exc:
        raise DataContractError(f"cannot read dataset registry: {path}") from exc
    if not isinstance(registry, dict):
        raise DataContractError("dataset registry must be a JSON object")
    if registry.get("schema_version") != "1.0" or not isinstance(registry.get("datasets"), list):
        raise DataContractError("dataset registry schema is incompatible")
    identifiers: list[str] = []
    for item in registry["datasets"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"].strip()
            or item["id"] != item["id"].strip()
        ):
            raise DataContractError("every registry dataset requires a non-empty trimmed string id")
        identifiers.append(item["id"])
        if item.get("kind") == "controlled_generator":
            if item.get("source") != "generated://finops-r/tradeops-sim":
                raise DataContractError(
                    "the core generator requires its canonical generated source"
                )
        elif not isinstance(item.get("official_url"), str) or not item["official_url"].startswith(
            "https://"
        ):
            raise DataContractError(f"registry dataset {item['id']} requires an HTTPS official_url")
        if item.get("role") not in {"core", "proxy", "stress", "optional"}:
            raise DataContractError(f"registry dataset {item['id']} has an invalid role")
    if len(identifiers) != len(set(identifiers)):
        raise DataContractError("dataset registry ids must be unique")
    return registry
