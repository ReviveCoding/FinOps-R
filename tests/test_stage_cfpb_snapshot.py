from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage_cfpb_snapshot import CFPB_COLUMNS, CfpbContractError, stage_cfpb_snapshot


def _rows(count: int = 12) -> list[list[str]]:
    rows: list[list[str]] = []
    for index in range(count):
        received_day = index + 1
        sent_day = received_day + 1
        rows.append(
            [
                f"01/{received_day:02d}/24",
                "Credit card" if index % 2 else "Debt collection",
                "General-purpose credit card",
                "Fees or interest" if index % 2 else "Attempts to collect debt not owed",
                "Problem with fees",
                f"Complete public complaint narrative {index}",
                "None",
                "Example Company",
                "NJ",
                "089XX",
                "None",
                "Consent provided",
                "Web",
                f"01/{sent_day:02d}/24",
                "Closed with explanation",
                "Yes",
                "N/A",
                str(1_000_000 + index),
            ]
        )
    return rows


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(CFPB_COLUMNS)
        writer.writerows(rows)


class CfpbStagingTests(unittest.TestCase):
    def test_valid_snapshot_stages_atomically_with_temporal_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            output = root / "staged"
            _write_csv(source, _rows())
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = stage_cfpb_snapshot(
                source,
                output,
                expected_sha256=source_hash,
                expected_rows=12,
                min_rows=10,
                source_archive_name="snapshot.7z",
            )
            self.assertEqual(manifest["row_count"], 12)
            self.assertEqual(manifest["unique_complaint_ids"], 12)
            self.assertEqual(
                set(manifest["temporal_split_counts"]), {"train", "validation", "test"}
            )
            self.assertEqual(
                hashlib.sha256((output / "complaints.csv").read_bytes()).hexdigest(), source_hash
            )
            self.assertTrue((output / "dataset_manifest.json").is_file())
            self.assertTrue((output / "label_taxonomy.json").is_file())
            self.assertTrue((output / "temporal_splits.csv").is_file())
            with self.assertRaises(FileExistsError):
                stage_cfpb_snapshot(source, output, min_rows=10)

    def test_truncated_record_is_rejected_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            output = root / "staged"
            _write_csv(source, _rows())
            with source.open("a", encoding="utf-8", newline="") as stream:
                stream.write("05/10/24,Student loan,Private student loan,Problem wi")
            with self.assertRaisesRegex(CfpbContractError, "expected 18 fields"):
                stage_cfpb_snapshot(source, output, min_rows=10)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".staged-*")), [])

    def test_duplicate_id_invalid_date_and_hash_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            duplicate = _rows()
            duplicate[1][-1] = duplicate[0][-1]
            cases.append(("duplicate", duplicate, "duplicate Complaint ID"))
            bad_date = _rows()
            bad_date[1][0] = "2024-01-02"
            cases.append(("bad-date", bad_date, "Date received"))
            for name, rows, message in cases:
                source = root / f"{name}.csv"
                output = root / f"{name}-staged"
                _write_csv(source, rows)
                with self.subTest(name=name), self.assertRaisesRegex(CfpbContractError, message):
                    stage_cfpb_snapshot(source, output, min_rows=10)
                self.assertFalse(output.exists())

            source = root / "hash.csv"
            _write_csv(source, _rows())
            with self.assertRaisesRegex(CfpbContractError, "SHA-256 mismatch"):
                stage_cfpb_snapshot(
                    source, root / "hash-staged", expected_sha256="0" * 64, min_rows=10
                )

    def test_exact_repair_evidence_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            output = root / "staged"
            quarantine = root / "truncated.bin"
            repair_path = root / "repair.json"
            _write_csv(source, _rows())
            quarantine.write_bytes(b"truncated final record")
            clean_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            quarantine_hash = hashlib.sha256(quarantine.read_bytes()).hexdigest()
            repair = {
                "repair_type": "remove_exact_truncated_final_record",
                "original_archive_bytes": 123,
                "original_archive_sha256": "1" * 64,
                "original_csv_bytes": source.stat().st_size + quarantine.stat().st_size,
                "original_csv_sha256": "2" * 64,
                "clean_csv_bytes": source.stat().st_size,
                "clean_csv_sha256": clean_hash,
                "quarantined_bytes": quarantine.stat().st_size,
                "quarantined_sha256": quarantine_hash,
            }
            repair_path.write_text(json.dumps(repair), encoding="utf-8")
            manifest = stage_cfpb_snapshot(
                source,
                output,
                expected_sha256=clean_hash,
                expected_rows=12,
                min_rows=10,
                repair_report_path=repair_path,
                quarantine_path=quarantine,
            )
            self.assertEqual(manifest["repair"], repair)
            self.assertEqual(
                (output / "quarantine" / "truncated_final_record.bin").read_bytes(),
                quarantine.read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
