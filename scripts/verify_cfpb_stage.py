from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_CSV_BYTES = 1_882_740_229
EXPECTED_CSV_SHA256 = "7e67e619c39baecc75d41091cb7a6d4b9f9e9fd700e0d406c9cc1f3a7156b8e1"
EXPECTED_QUARANTINE_BYTES = 90
EXPECTED_QUARANTINE_SHA256 = "9e304012965a8948e1edda00f0f0ed475bc4788a54abdb0b2875fc214e5023df"
EXPECTED_ROWS = 1_406_112
EXPECTED_FILES = {
    "README.md",
    "complaints.csv",
    "dataset_manifest.json",
    "label_taxonomy.json",
    "quarantine",
    "quarantine/truncated_final_record.bin",
    "source_repair.json",
    "temporal_splits.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> dict[str, object]:
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"staged root must be one regular directory: {root}")
    observed: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        observed.add(relative)
        if path.is_symlink():
            raise ValueError(f"staged output contains a symlink: {relative}")
    if observed != EXPECTED_FILES:
        raise ValueError(
            f"staged file set mismatch; missing={sorted(EXPECTED_FILES - observed)}, "
            f"unexpected={sorted(observed - EXPECTED_FILES)}"
        )

    complaints = root / "complaints.csv"
    if complaints.stat().st_size != EXPECTED_CSV_BYTES:
        raise ValueError("staged complaint CSV byte count mismatch")
    if _sha256(complaints) != EXPECTED_CSV_SHA256:
        raise ValueError("staged complaint CSV SHA-256 mismatch")

    quarantine = root / "quarantine" / "truncated_final_record.bin"
    if quarantine.stat().st_size != EXPECTED_QUARANTINE_BYTES:
        raise ValueError("quarantine byte count mismatch")
    if _sha256(quarantine) != EXPECTED_QUARANTINE_SHA256:
        raise ValueError("quarantine SHA-256 mismatch")

    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    repair = json.loads((root / "source_repair.json").read_text(encoding="utf-8"))
    taxonomy = json.loads((root / "label_taxonomy.json").read_text(encoding="utf-8"))
    if manifest["dataset_id"] != "cfpb_consumer_complaints":
        raise ValueError("dataset identity mismatch")
    if manifest["row_count"] != EXPECTED_ROWS or manifest["unique_complaint_ids"] != EXPECTED_ROWS:
        raise ValueError("manifest row or identity count mismatch")
    if manifest["source_sha256"] != EXPECTED_CSV_SHA256:
        raise ValueError("manifest source SHA-256 mismatch")
    if manifest["repair"] != repair:
        raise ValueError("manifest and standalone repair evidence disagree")
    if repair["clean_csv_sha256"] != EXPECTED_CSV_SHA256:
        raise ValueError("repair clean CSV SHA-256 mismatch")
    if repair["quarantined_sha256"] != EXPECTED_QUARANTINE_SHA256:
        raise ValueError("repair quarantine SHA-256 mismatch")

    split_counts = manifest["temporal_split_counts"]
    if set(split_counts) != {"train", "validation", "test"}:
        raise ValueError("manifest chronological split keys mismatch")
    if sum(split_counts.values()) != EXPECTED_ROWS:
        raise ValueError("manifest chronological split counts do not sum to row count")
    if sum(taxonomy["product_counts"].values()) != EXPECTED_ROWS:
        raise ValueError("product taxonomy counts do not sum to row count")
    if sum(taxonomy["issue_counts"].values()) != EXPECTED_ROWS:
        raise ValueError("issue taxonomy counts do not sum to row count")

    split_path = root / "temporal_splits.csv"
    observed_splits: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    seen_ids: set[str] = set()
    with split_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != ["Complaint ID", "Date received", "split"]:
            raise ValueError("temporal split schema mismatch")
        for row in reader:
            complaint_id = row["Complaint ID"]
            split = row["split"]
            if complaint_id in seen_ids:
                raise ValueError(f"duplicate ID in temporal splits: {complaint_id}")
            if split not in observed_splits:
                raise ValueError(f"unexpected split label: {split}")
            seen_ids.add(complaint_id)
            observed_splits[split] += 1
    if len(seen_ids) != EXPECTED_ROWS:
        raise ValueError("temporal split identity count mismatch")
    if observed_splits != split_counts:
        raise ValueError("temporal split file and manifest counts disagree")

    return {
        "status": "PASS",
        "row_count": EXPECTED_ROWS,
        "source_sha256": EXPECTED_CSV_SHA256,
        "quarantined_bytes": EXPECTED_QUARANTINE_BYTES,
        "temporal_split_counts": observed_splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the exact staged CFPB Phase 2 snapshot")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
