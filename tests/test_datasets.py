from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from finops_r.datasets import (
    DataContractError,
    _copy_bounded_archive,
    inspect_zip,
    load_dataset_registry,
    stage_iso20022,
    validate_iso20022_frames,
)


def _iso_frames() -> dict[str, pd.DataFrame]:
    analytics = pd.DataFrame(
        [
            {
                "MessageId": f"MSG-{index}",
                "EndToEndId": f"E2E-{index}",
                "TxDateTime": f"2025-01-{index + 1:02d}T10:00:00",
                "StatusTimestamp": f"2025-01-{index + 1:02d}T10:00:0{index % 9 + 1}",
                "ProcessingTimeSecs": index % 9 + 1,
                "IntrBkSttlmDt": f"2025-01-{index + 1:02d}",
                "InstdAmt": 100.0 + index,
                "Currency": "USD",
                "DbtrAgtBIC": "AAAAUS00",
                "CdtrAgtBIC": "BBBBGB00",
                "DbtrCountry": "US",
                "CdtrCountry": "GB",
                "PurposeCode": "SUPP",
                "TxSts": "RJCT" if index in {2, 7} else "ACSP",
                "RejectionReason": "AC01" if index in {2, 7} else None,
            }
            for index in range(10)
        ]
    )
    pacs002 = analytics[["MessageId", "TxSts", "RejectionReason", "StatusTimestamp"]].rename(
        columns={"MessageId": "OriginalMessageId"}
    )
    pacs008 = analytics[
        [
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
        ]
    ]
    return {
        "iso20022_analytics.csv": analytics,
        "pacs002.csv": pacs002,
        "pacs008.csv": pacs008,
    }


def _write_iso_zip(path: Path, frames: dict[str, pd.DataFrame] | None = None) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, frame in (frames or _iso_frames()).items():
            bundle.writestr(name, frame.to_csv(index=False))


class DatasetTests(unittest.TestCase):
    def test_archive_snapshot_copy_is_bounded_before_zip_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            destination = root / "snapshot.zip"
            source.write_bytes(b"1234")
            with self.assertRaisesRegex(DataContractError, "exceeds"):
                _copy_bounded_archive(source, destination, 3)

    def test_iso_snapshot_stages_atomically_with_temporal_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "iso.zip"
            output = root / "staged"
            _write_iso_zip(archive)
            manifest = stage_iso20022(archive, output)
            self.assertEqual(manifest["validation"]["status"], "PASS")
            self.assertEqual(
                manifest["temporal_split_counts"], {"train": 6, "test": 2, "validation": 2}
            )
            self.assertFalse(manifest["official_iso_dataset"])
            self.assertTrue((output / "dataset_manifest.json").is_file())
            assignments = pd.read_csv(output / "temporal_splits.csv")
            self.assertEqual(set(assignments["split"]), {"train", "validation", "test"})
            with self.assertRaises(FileExistsError):
                stage_iso20022(archive, output)

    def test_iso_pair_status_mismatch_is_rejected_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            output = root / "staged"
            frames = _iso_frames()
            frames["pacs002.csv"].loc[0, "TxSts"] = "RJCT"
            _write_iso_zip(archive, frames)
            with self.assertRaisesRegex(DataContractError, "disagree on TxSts"):
                stage_iso20022(archive, output)
            self.assertFalse(output.exists())

    def test_iso_non_finite_amount_is_rejected(self) -> None:
        frames = _iso_frames()
        frames["iso20022_analytics.csv"].loc[0, "InstdAmt"] = float("inf")
        frames["pacs008.csv"].loc[0, "InstdAmt"] = float("inf")
        with self.assertRaisesRegex(DataContractError, "finite and positive"):
            validate_iso20022_frames(frames)

    def test_iso_end_to_end_identity_must_be_complete_and_unique(self) -> None:
        for invalid_value, message in (("", "non-empty"), ("E2E-1", "unique")):
            frames = _iso_frames()
            frames["iso20022_analytics.csv"].loc[0, "EndToEndId"] = invalid_value
            frames["pacs008.csv"].loc[0, "EndToEndId"] = invalid_value
            with (
                self.subTest(invalid_value=invalid_value),
                self.assertRaisesRegex(DataContractError, message),
            ):
                validate_iso20022_frames(frames)

    def test_iso_staging_is_removed_on_keyboard_interrupt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "iso.zip"
            output = root / "staged"
            _write_iso_zip(archive)
            with patch.object(Path, "write_bytes", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    stage_iso20022(archive, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".staged-*")), [])

    def test_iso_staging_uses_one_immutable_archive_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "iso.zip"
            replacement = root / "replacement.zip"
            output = root / "staged"
            original_frames = _iso_frames()
            replacement_frames = _iso_frames()
            replacement_frames["iso20022_analytics.csv"].loc[0, "InstdAmt"] = 999.0
            replacement_frames["pacs008.csv"].loc[0, "InstdAmt"] = 999.0
            _write_iso_zip(archive, original_frames)
            _write_iso_zip(replacement, replacement_frames)
            original_sha = hashlib.sha256(archive.read_bytes()).hexdigest()

            from finops_r.datasets import inspect_zip as real_inspect_zip

            def replace_source_after_inspection(path: Path, **kwargs: object) -> dict[str, object]:
                report = real_inspect_zip(path, **kwargs)
                archive.write_bytes(replacement.read_bytes())
                return report

            with patch(
                "finops_r.datasets.inspect_zip", side_effect=replace_source_after_inspection
            ):
                manifest = stage_iso20022(archive, output)

            staged = pd.read_csv(output / "iso20022_analytics.csv")
            self.assertEqual(float(staged.loc[0, "InstdAmt"]), 100.0)
            self.assertEqual(manifest["source_sha256"], original_sha)

    def test_zip_traversal_and_duplicate_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traversal = root / "traversal.zip"
            with zipfile.ZipFile(traversal, "w") as bundle:
                bundle.writestr("../escape.txt", "unsafe")
            with self.assertRaisesRegex(DataContractError, "unsafe member"):
                inspect_zip(traversal)

            duplicate = root / "duplicate.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(duplicate, "w") as bundle:
                    bundle.writestr("same.txt", "one")
                    bundle.writestr("same.txt", "two")
            with self.assertRaisesRegex(DataContractError, "duplicate member"):
                inspect_zip(duplicate)

            special = root / "special.zip"
            with zipfile.ZipFile(special, "w") as bundle:
                member = zipfile.ZipInfo("named-pipe")
                member.create_system = 3
                member.external_attr = (stat.S_IFIFO | 0o600) << 16
                bundle.writestr(member, b"payload")
            with self.assertRaisesRegex(DataContractError, "special"):
                inspect_zip(special)

    def test_zip_rejects_cross_platform_unsafe_names_and_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, name in enumerate(("C:/escape.txt", "AUX.txt")):
                archive = root / f"unsafe-{index}.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr(name, "unsafe")
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(DataContractError, "unsafe member"),
                ):
                    inspect_zip(archive)

            collision_sets = (
                ("Data.csv", "data.csv"),
                (
                    "caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt",
                    "cafe\N{COMBINING ACUTE ACCENT}.txt",
                ),
                ("node", "node/child.txt"),
            )
            for index, names in enumerate(collision_sets):
                archive = root / f"collision-{index}.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    for name in names:
                        bundle.writestr(name, "payload")
                with (
                    self.subTest(names=names),
                    self.assertRaisesRegex(DataContractError, "collision"),
                ):
                    inspect_zip(archive)

    def test_registry_is_machine_readable_and_unique(self) -> None:
        registry_path = Path(__file__).parents[1] / "src" / "finops_r" / "dataset_registry.json"
        registry = load_dataset_registry(registry_path)
        identifiers = [item["id"] for item in registry["datasets"]]
        self.assertIn("tradeops_sim", identifiers)
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_registry_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            item = {"id": "same", "role": "core", "official_url": "https://example.com"}
            path.write_text(
                json.dumps({"schema_version": "1.0", "datasets": [item, item]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DataContractError, "unique"):
                load_dataset_registry(path)

    def test_registry_rejects_malformed_top_level_and_empty_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            for value in (
                [],
                {"schema_version": "1.0", "datasets": [{"id": "", "role": "core"}]},
            ):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(value=value), self.assertRaises(DataContractError):
                    load_dataset_registry(path)


if __name__ == "__main__":
    unittest.main()
