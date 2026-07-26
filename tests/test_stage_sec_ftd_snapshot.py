from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


def load_module(name: str, filename: str):
    root = Path(__file__).resolve().parents[1]
    candidate = root / "scripts" / filename
    if not candidate.is_file():
        candidate = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, candidate)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage_module = load_module("stage_sec_ftd_snapshot", "stage_sec_ftd_snapshot.py")
verify_module = load_module("verify_sec_ftd_stage", "verify_sec_ftd_stage.py")


HEADER = [
    "SETTLEMENT DATE",
    "CUSIP",
    "SYMBOL",
    "QUANTITY (FAILS)",
    "DESCRIPTION",
    "PRICE",
]


class SecFtdStagingTests(unittest.TestCase):
    def make_zip(
        self,
        raw_dir: Path,
        name: str,
        rows,
        *,
        member=None,
        header=None,
        trailer_count=None,
        trailer_quantity=None,
        include_trailer=True,
    ):
        member = member or name.replace(".zip", ".txt")
        rows = list(rows)
        stream = io.StringIO(newline="")
        stream.write("|".join(HEADER if header is None else header) + "\n")
        for row in rows:
            stream.write("|".join(str(value) for value in row) + "\n")
        if include_trailer:
            record_count = len(rows) if trailer_count is None else trailer_count
            if trailer_quantity is None:
                trailer_quantity = sum(int(row[3]) for row in rows if str(row[3]).isdigit())
            stream.write(f"Trailer record count {record_count}\n")
            stream.write(f"Trailer total quantity of shares {trailer_quantity}\n")
        info = zipfile.ZipInfo(member, date_time=(2020, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        with zipfile.ZipFile(raw_dir / name, "w") as archive:
            archive.writestr(info, stream.getvalue().encode("utf-8"))

    def valid_raw(self, root: Path):
        raw = root / "raw"
        raw.mkdir()
        self.make_zip(
            raw,
            "cnsfails202001a.zip",
            [
                ["20200102", "111111111", "AAA", "10", "ALPHA INC", "2.50"],
                ["20200103", "222222222", "BBB", "7", "BETA INC", "."],
            ],
        )
        self.make_zip(
            raw,
            "cnsfails202001b.zip",
            [
                ["20200116", "111111111", "AAA", "4", "ALPHA INC", "3"],
                ["20200117", "333333333", "CCC", "2", "GAMMA INC", "1.25"],
            ],
        )
        return raw

    def stage(self, raw: Path, output: Path):
        return stage_module.stage(raw, output, "202001", "202001")

    def test_valid_snapshot_stages_atomically_and_verifies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            output = root / "stage"
            manifest = self.stage(raw, output)
            self.assertEqual(manifest["archive_count"], 2)
            self.assertEqual(manifest["counts"]["fails_rows"], 4)
            result = verify_module.verify(output, raw)
            self.assertEqual(result["file_integrity"], "PASS")
            self.assertEqual(result["fails_rows"], 4)

    def test_missing_archive_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            (raw / "cnsfails202001b.zip").unlink()
            output = root / "stage"
            with self.assertRaises(stage_module.ContractError):
                self.stage(raw, output)
            self.assertFalse(output.exists())

    def test_unsafe_zip_member_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            self.make_zip(
                raw,
                "cnsfails202001a.zip",
                [["20200102", "111111111", "AAA", "10", "ALPHA INC", "2"]],
                member="../escape.txt",
            )
            output = root / "stage"
            with self.assertRaises(stage_module.ContractError):
                self.stage(raw, output)
            self.assertFalse(output.exists())

    def test_date_outside_archive_half_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            self.make_zip(
                raw,
                "cnsfails202001a.zip",
                [["20200116", "111111111", "AAA", "10", "ALPHA INC", "2"]],
            )
            with self.assertRaises(stage_module.ContractError):
                self.stage(raw, root / "stage")

    def test_source_half_boundary_uses_days_1_to_14_and_15_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            self.make_zip(
                raw,
                "cnsfails202001a.zip",
                [["20200114", "111111111", "AAA", "10", "ALPHA INC", "2"]],
            )
            self.make_zip(
                raw,
                "cnsfails202001b.zip",
                [["20200115", "222222222", "BBB", "7", "BETA INC", "3"]],
            )
            manifest = self.stage(raw, root / "stage")
            self.assertEqual(manifest["counts"]["fails_rows"], 2)

    def test_source_routes_match_sec_download_exceptions(self):
        cases = [
            ("202002", "/files/node/add/data_distribution/"),
            ("202605", "/files/data/other/fails-deliver-data/"),
            ("202606", "/files/data/fails-deliver-data/"),
        ]
        for month, fragment in cases:
            with self.subTest(month=month):
                source = stage_module.source_specs(month, month)[0]
                self.assertIn(fragment, source.url)
                self.assertEqual(source.url, verify_module.source_url(source.name, source.month))

    def test_source_member_name_variants_are_frozen(self):
        self.assertEqual(
            stage_module.member_name_class("cnsfails202205b.zip", "cnsfails202205b"),
            "expected_extensionless",
        )
        self.assertEqual(
            stage_module.member_name_class(
                "cnsfails202204b.zip",
                "SEC Failed To Deliver April 2022 second half.txt",
            ),
            "known_exception",
        )
        self.assertEqual(
            stage_module.member_name_class("cnsfails202308b.zip", "cnsfails202309a.txt"),
            "known_exception",
        )
        self.assertEqual(
            stage_module.member_name_class("cnsfails202308b.zip", "cnsfails202309a.txt"),
            verify_module.member_name_class("cnsfails202308b.zip", "cnsfails202309a.txt"),
        )
        with self.assertRaises(stage_module.ContractError):
            stage_module.member_name_class("cnsfails202001a.zip", "unrelated.txt")

    def test_official_symbol_exceptions_are_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            self.make_zip(
                raw,
                "cnsfails202001a.zip",
                [
                    [
                        "20200102",
                        "111111111",
                        "3126REORGPAY",
                        "10",
                        "ALPHA INC",
                        "2",
                    ],
                    ["20200103", "222222222", "", "7", "BETA INC", "3"],
                ],
            )
            output = root / "stage"
            manifest = self.stage(raw, output)
            self.assertEqual(manifest["source_quality"]["symbol_over_official_limit_rows"], 1)
            self.assertEqual(manifest["source_quality"]["blank_symbol_rows"], 1)
            self.assertEqual(manifest["source_quality"]["max_symbol_length"], 12)
            self.assertEqual(verify_module.verify(output, raw)["file_integrity"], "PASS")

    def test_unescaped_description_pipe_is_repaired_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            self.make_zip(
                raw,
                "cnsfails202001a.zip",
                [
                    [
                        "20200102",
                        "23344K110",
                        "DMYQWS",
                        "351",
                        "DMY TECHNOLOGY GROUP INC IV | ",
                        "1.49",
                    ]
                ],
            )
            output = root / "stage"
            manifest = self.stage(raw, output)
            self.assertEqual(manifest["source_quality"]["delimiter_repaired_rows"], 1)
            self.assertEqual(verify_module.verify(output, raw)["file_integrity"], "PASS")

    def test_missing_or_mismatched_trailer_is_rejected(self):
        variants = [
            {"include_trailer": False},
            {"trailer_count": 99},
            {"trailer_quantity": 99},
        ]
        for options in variants:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                raw = self.valid_raw(root)
                self.make_zip(
                    raw,
                    "cnsfails202001a.zip",
                    [["20200102", "111111111", "AAA", "10", "ALPHA INC", "2"]],
                    **options,
                )
                with self.assertRaises(stage_module.ContractError):
                    self.stage(raw, root / "stage")

    def test_duplicate_settlement_date_cusip_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            self.make_zip(
                raw,
                "cnsfails202001a.zip",
                [
                    ["20200102", "111111111", "AAA", "10", "ALPHA INC", "2"],
                    ["20200102", "111111111", "AAA", "11", "ALPHA INC", "2"],
                ],
            )
            with self.assertRaises(stage_module.ContractError):
                self.stage(raw, root / "stage")

    def test_reverse_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            self.make_zip(
                raw,
                "cnsfails202001a.zip",
                [
                    ["20200103", "111111111", "AAA", "10", "ALPHA INC", "2"],
                    ["20200102", "222222222", "BBB", "11", "BETA INC", "2"],
                ],
            )
            with self.assertRaises(stage_module.ContractError):
                self.stage(raw, root / "stage")

    def test_malformed_quantity_and_price_are_rejected(self):
        bad_rows = [
            ["20200102", "111111111", "AAA", "-1", "ALPHA INC", "2"],
            ["20200102", "111111111", "AAA", "1", "ALPHA INC", "NaN"],
        ]
        for row in bad_rows:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                raw = self.valid_raw(root)
                self.make_zip(raw, "cnsfails202001a.zip", [row])
                with self.assertRaises(stage_module.ContractError):
                    self.stage(raw, root / "stage")

    def test_verifier_rejects_published_file_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            output = root / "stage"
            self.stage(raw, output)
            with (output / "daily_stress.csv").open("a", encoding="utf-8") as handle:
                handle.write("tamper\n")
            with self.assertRaises(verify_module.VerificationError):
                verify_module.verify(output, raw)

    def test_verifier_rejects_source_archive_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            output = root / "stage"
            self.stage(raw, output)
            with (raw / "cnsfails202001a.zip").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(verify_module.VerificationError):
                verify_module.verify(output, raw)

    def test_verifier_rejects_manifest_source_quality_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            output = root / "stage"
            self.stage(raw, output)
            manifest_path = output / "manifest.json"
            manifest = verify_module.load_manifest(manifest_path)
            manifest["source_quality"]["blank_symbol_rows"] += 1
            stage_module.write_json(manifest_path, manifest)
            with self.assertRaises(verify_module.VerificationError):
                verify_module.verify(output, raw)

    def test_outputs_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.valid_raw(root)
            first = root / "first"
            second = root / "second"
            self.stage(raw, first)
            self.stage(raw, second)
            for name in ("fails.csv.gz", "daily_stress.csv", "manifest.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
