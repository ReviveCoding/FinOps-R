from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_FILES = {"applications.csv", "events.csv.gz", "offers.csv", "manifest.json"}
EXPECTED_COUNTS = {
    "applications": 31_509,
    "events": 1_202_267,
    "offers": 42_995,
    "originators": 149,
}


class VerificationError(ValueError):
    """Raised when a staged BPI 2017 directory is invalid."""


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError("manifest.json is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise VerificationError("manifest root must be an object")
    if value.get("dataset") != "BPI Challenge 2017":
        raise VerificationError("unexpected dataset identity")
    if value.get("doi") != "10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b":
        raise VerificationError("unexpected dataset DOI")
    if value.get("counts") != EXPECTED_COUNTS:
        raise VerificationError(f"unexpected manifest counts: {value.get('counts')!r}")
    privacy = value.get("privacy")
    if not isinstance(privacy, dict) or privacy.get("raw_org_resource_published") is not False:
        raise VerificationError("manifest privacy contract is missing")
    return value


def verify_stage(stage: Path) -> dict[str, Any]:
    stage = stage.resolve()
    if stage.is_symlink() or not stage.is_dir():
        raise VerificationError(f"stage must be one real directory: {stage}")
    entries = list(stage.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise VerificationError("stage may contain regular files only")
    actual_files = {path.name for path in entries}
    if actual_files != EXPECTED_FILES:
        raise VerificationError(
            f"member set mismatch: expected {sorted(EXPECTED_FILES)}, got {sorted(actual_files)}"
        )
    manifest = _manifest(stage / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != EXPECTED_FILES - {"manifest.json"}:
        raise VerificationError("manifest file inventory is incompatible")
    for name, record in files.items():
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            raise VerificationError(f"invalid file record for {name}")
        size, digest = _sha256(stage / name)
        if record != {"bytes": size, "sha256": digest}:
            raise VerificationError(f"file integrity mismatch for {name}")

    application_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    application_event_counts: dict[str, int] = {}
    application_offer_counts: dict[str, int] = {}
    start_ranges: dict[str, list[str]] = {key: [] for key in ("train", "validation", "test")}
    with (stage / "applications.csv").open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_header = [
            "application_id",
            "case_start_utc",
            "case_end_utc",
            "event_count",
            "offer_count",
            "terminal_activity",
            "application_attributes_json",
            "split",
        ]
        if reader.fieldnames != expected_header:
            raise VerificationError("applications.csv header mismatch")
        for row_number, row in enumerate(reader, start=2):
            application_id = row["application_id"]
            if not application_id or application_id in application_ids:
                raise VerificationError(f"applications.csv row {row_number}: invalid duplicate ID")
            application_ids.add(application_id)
            split = row["split"]
            if split not in split_counts and split not in {
                "train",
                "validation",
                "test",
            }:
                raise VerificationError(f"applications.csv row {row_number}: invalid split")
            try:
                event_count = int(row["event_count"])
                offer_count = int(row["offer_count"])
                attrs = json.loads(row["application_attributes_json"])
            except (ValueError, json.JSONDecodeError) as exc:
                raise VerificationError(
                    f"applications.csv row {row_number}: invalid numeric or JSON value"
                ) from exc
            if event_count < 1 or offer_count < 0 or not isinstance(attrs, dict):
                raise VerificationError(f"applications.csv row {row_number}: invalid counts")
            application_event_counts[application_id] = event_count
            application_offer_counts[application_id] = offer_count
            split_counts[split] += 1
            start_ranges[split].append(row["case_start_utc"])
    if len(application_ids) != EXPECTED_COUNTS["applications"]:
        raise VerificationError("applications.csv count mismatch")
    if max(start_ranges["train"]) >= min(start_ranges["validation"]):
        raise VerificationError("train and validation timestamps overlap")
    if max(start_ranges["validation"]) >= min(start_ranges["test"]):
        raise VerificationError("validation and test timestamps overlap")
    if dict(split_counts) != manifest["split"]["counts"]:
        raise VerificationError("split counts do not match manifest")

    observed_event_counts: Counter[str] = Counter()
    event_offer_ids: dict[str, set[str]] = {}
    last_index: dict[str, int] = {}
    event_rows = 0
    with gzip.open(stage / "events.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_header = [
            "application_id",
            "event_index",
            "timestamp_utc",
            "activity",
            "event_origin",
            "lifecycle_transition",
            "offer_id",
            "resource_pseudonym",
            "attributes_json",
        ]
        if reader.fieldnames != expected_header:
            raise VerificationError("events.csv.gz header mismatch")
        for row_number, row in enumerate(reader, start=2):
            application_id = row["application_id"]
            if application_id not in application_ids:
                raise VerificationError(f"events row {row_number}: unknown application")
            try:
                index = int(row["event_index"])
                attrs = json.loads(row["attributes_json"])
            except (ValueError, json.JSONDecodeError) as exc:
                raise VerificationError(f"events row {row_number}: invalid value") from exc
            if index != last_index.get(application_id, 0) + 1:
                raise VerificationError(
                    f"events row {row_number}: event indexes are not contiguous"
                )
            last_index[application_id] = index
            if not isinstance(attrs, dict) or not row["activity"] or not row["timestamp_utc"]:
                raise VerificationError(f"events row {row_number}: required value is missing")
            resource = row["resource_pseudonym"]
            if resource and not resource.startswith("resource_"):
                raise VerificationError(f"events row {row_number}: raw resource may be present")
            if attrs.get("org:resource", "") != resource:
                raise VerificationError(f"events row {row_number}: resource projection mismatch")
            offer_id = row["offer_id"]
            if offer_id:
                event_offer_ids.setdefault(application_id, set()).add(offer_id)
            observed_event_counts[application_id] += 1
            event_rows += 1
    if event_rows != EXPECTED_COUNTS["events"]:
        raise VerificationError("events.csv.gz count mismatch")
    if dict(observed_event_counts) != application_event_counts:
        raise VerificationError("per-application event counts do not reconcile")

    offer_ids: set[str] = set()
    observed_offer_counts: Counter[str] = Counter()
    with (stage / "offers.csv").open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_header = [
            "application_id",
            "offer_id",
            "first_timestamp_utc",
            "last_timestamp_utc",
            "event_count",
            "terminal_activity",
        ]
        if reader.fieldnames != expected_header:
            raise VerificationError("offers.csv header mismatch")
        for row_number, row in enumerate(reader, start=2):
            application_id = row["application_id"]
            offer_id = row["offer_id"]
            if application_id not in application_ids or not offer_id or offer_id in offer_ids:
                raise VerificationError(f"offers row {row_number}: invalid application or offer ID")
            if offer_id not in event_offer_ids.get(application_id, set()):
                raise VerificationError(f"offers row {row_number}: no supporting event")
            offer_ids.add(offer_id)
            observed_offer_counts[application_id] += 1
    if len(offer_ids) != EXPECTED_COUNTS["offers"]:
        raise VerificationError("offers.csv count mismatch")
    if dict(observed_offer_counts) != {
        key: value for key, value in application_offer_counts.items() if value
    }:
        raise VerificationError("per-application offer counts do not reconcile")

    return {
        "applications": len(application_ids),
        "events": event_rows,
        "offers": len(offer_ids),
        "split_counts": dict(split_counts),
        "file_integrity": "PASS",
        "referential_integrity": "PASS",
        "privacy_contract": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a staged BPI Challenge 2017 snapshot")
    parser.add_argument("--stage", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify_stage(args.stage)
    except Exception as exc:
        print(f"BPI2017_VERIFICATION_ERROR={type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    print("BPI2017_STAGE_VERIFICATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
