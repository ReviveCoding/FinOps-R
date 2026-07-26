from __future__ import annotations

import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.stage_bpi2017_snapshot import BpiContractError, stage_bpi2017_snapshot
from scripts import verify_bpi2017_stage as verifier


def _trace(
    application_id: str,
    day: int,
    *,
    offer_id: str = "",
    reverse: bool = False,
) -> str:
    first_hour, second_hour = (1, 0) if reverse else (0, 1)
    offer = f'<string key="OfferID" value="{offer_id}"/>' if offer_id else ""
    return f"""
    <trace>
      <string key="concept:name" value="{application_id}"/>
      <string key="ApplicationType" value="New credit"/>
      <float key="RequestedAmount" value="10000"/>
      <event>
        <string key="concept:name" value="A_Create Application"/>
        <date key="time:timestamp" value="2016-01-{day:02d}T{first_hour:02d}:00:00+00:00"/>
        <string key="EventOrigin" value="Application"/>
        <string key="org:resource" value="employee-A"/>
      </event>
      <event>
        <string key="concept:name" value="{"O_Create Offer" if offer_id else "A_Submitted"}"/>
        <date key="time:timestamp" value="2016-01-{day:02d}T{second_hour:02d}:00:00+00:00"/>
        <string key="EventOrigin" value="{"Offer" if offer_id else "Application"}"/>
        <string key="org:resource" value="employee-B"/>
        {offer}
      </event>
    </trace>
    """


def _write_xes(path: Path, traces: list[str]) -> tuple[int, str]:
    document = (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<log xmlns="http://www.xes-standard.org/">\n' + "\n".join(traces) + "\n</log>\n"
    )
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as stream:
        stream.write(document.encode())
    payload = path.read_bytes()
    return len(payload), hashlib.md5(payload).hexdigest()


def _stage(source: Path, output: Path) -> dict[str, object]:
    size = source.stat().st_size
    md5 = hashlib.md5(source.read_bytes()).hexdigest()
    return stage_bpi2017_snapshot(
        source,
        output,
        expected_bytes=size,
        expected_md5=md5,
        expected_applications=4,
        expected_events=8,
        expected_offers=2,
        expected_originators=2,
    )


class Bpi2017StagingTests(unittest.TestCase):
    def test_valid_multi_offer_log_stages_atomically_and_redacts_resources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "BPI Challenge 2017.xes.gz"
            output = root / "staged"
            _write_xes(
                source,
                [
                    _trace("Application_1", 1, offer_id="Offer_1"),
                    _trace("Application_2", 2, offer_id="Offer_2"),
                    _trace("Application_3", 3),
                    _trace("Application_4", 4),
                ],
            )
            manifest = _stage(source, output)
            self.assertEqual(
                manifest["counts"],
                {"applications": 4, "events": 8, "offers": 2, "originators": 2},
            )
            self.assertEqual(sum(manifest["split"]["counts"].values()), 4)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"applications.csv", "events.csv.gz", "offers.csv", "manifest.json"},
            )
            with gzip.open(output / "events.csv.gz", "rt", encoding="utf-8") as stream:
                published = stream.read()
            self.assertNotIn("employee-A", published)
            self.assertNotIn("employee-B", published)
            self.assertIn("resource_", published)
            with (output / "applications.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["split"] for row in rows}, {"train", "validation", "test"})
            json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    def test_duplicate_application_is_rejected_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "duplicate.xes.gz"
            output = root / "staged"
            _write_xes(
                source,
                [
                    _trace("Application_1", 1, offer_id="Offer_1"),
                    _trace("Application_1", 2, offer_id="Offer_2"),
                    _trace("Application_3", 3),
                    _trace("Application_4", 4),
                ],
            )
            with self.assertRaisesRegex(BpiContractError, "duplicate application ID"):
                _stage(source, output)
            self.assertFalse(output.exists())

    def test_reverse_timestamp_is_rejected_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reverse.xes.gz"
            output = root / "staged"
            _write_xes(
                source,
                [
                    _trace("Application_1", 1, offer_id="Offer_1", reverse=True),
                    _trace("Application_2", 2, offer_id="Offer_2"),
                    _trace("Application_3", 3),
                    _trace("Application_4", 4),
                ],
            )
            with self.assertRaisesRegex(BpiContractError, "timestamps are not nondecreasing"):
                _stage(source, output)
            self.assertFalse(output.exists())

    def test_archive_hash_mismatch_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.xes.gz"
            output = root / "staged"
            size, _ = _write_xes(source, [_trace("Application_1", 1)])
            with self.assertRaisesRegex(BpiContractError, "archive MD5 mismatch"):
                stage_bpi2017_snapshot(
                    source,
                    output,
                    expected_bytes=size,
                    expected_md5="0" * 32,
                    expected_applications=1,
                    expected_events=2,
                    expected_offers=1,
                    expected_originators=2,
                )
            self.assertFalse(output.exists())

    def test_independent_verifier_reconciles_synthetic_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.xes.gz"
            output = root / "staged"
            _write_xes(
                source,
                [
                    _trace("Application_1", 1, offer_id="Offer_1"),
                    _trace("Application_2", 2, offer_id="Offer_2"),
                    _trace("Application_3", 3),
                    _trace("Application_4", 4),
                ],
            )
            _stage(source, output)
            expected = {"applications": 4, "events": 8, "offers": 2, "originators": 2}
            with mock.patch.object(verifier, "EXPECTED_COUNTS", expected):
                result = verifier.verify_stage(output)
            self.assertEqual(result["referential_integrity"], "PASS")
            self.assertEqual(result["privacy_contract"], "PASS")

    def test_independent_verifier_rejects_file_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.xes.gz"
            output = root / "staged"
            _write_xes(
                source,
                [
                    _trace("Application_1", 1, offer_id="Offer_1"),
                    _trace("Application_2", 2, offer_id="Offer_2"),
                    _trace("Application_3", 3),
                    _trace("Application_4", 4),
                ],
            )
            _stage(source, output)
            with (output / "offers.csv").open("a", encoding="utf-8") as stream:
                stream.write("tamper\n")
            expected = {"applications": 4, "events": 8, "offers": 2, "originators": 2}
            with (
                mock.patch.object(verifier, "EXPECTED_COUNTS", expected),
                self.assertRaisesRegex(verifier.VerificationError, "file integrity mismatch"),
            ):
                verifier.verify_stage(output)


if __name__ == "__main__":
    unittest.main()
