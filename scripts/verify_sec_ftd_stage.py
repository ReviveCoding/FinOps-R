#!/usr/bin/env python3
"""Independent verifier for a staged SEC fails-to-deliver snapshot."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import zipfile
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

EXPECTED_FILES = {"daily_stress.csv", "fails.csv.gz", "manifest.json"}
FAILS_HEADER = (
    "settlement_date",
    "cusip",
    "symbol",
    "quantity_fails",
    "description",
    "previous_close_price",
    "fail_value_proxy_usd",
    "source_archive",
    "source_half",
    "split",
)
DAILY_HEADER = (
    "settlement_date",
    "securities_with_fails",
    "total_fail_shares",
    "priced_fail_shares",
    "fail_value_proxy_usd",
    "missing_price_rows",
    "missing_price_share",
    "split",
)
TRAIN_END = date(2023, 12, 31)
VALIDATION_END = date(2024, 12, 31)
REQUIRED_WARNINGS = {
    "fail_value_proxy",
    "missing_zero_semantics",
    "price_warning",
    "short_sale_warning",
    "stock_warning",
    "use",
}
DOWNLOAD_ROOT = "https://www.sec.gov/files/data/fails-deliver-data"
LEGACY_DOWNLOAD_ROOT = "https://www.sec.gov/files/node/add/data_distribution"
OTHER_DOWNLOAD_ROOT = "https://www.sec.gov/files/data/other/fails-deliver-data"
LEGACY_MONTHS = {"202002", "202003", "202004"}
OTHER_MONTHS = {"202605"}
KNOWN_MEMBER_ALIASES = {
    "cnsfails202204b.zip": "SEC Failed To Deliver April 2022 second half.txt",
    "cnsfails202308b.zip": "cnsfails202309a.txt",
}
OFFICIAL_MAX_SYMBOL_LENGTH = 10
SOURCE_SYMBOL_HARD_LIMIT = 32
TRAILER_RECORD_RE = re.compile(r"^Trailer record count (?P<value>\d+)$")
TRAILER_QUANTITY_RE = re.compile(r"^Trailer total quantity of shares (?P<value>\d+)$")


class VerificationError(RuntimeError):
    """Raised when the published stage does not satisfy its manifest."""


def reject_constant(value: str):
    raise VerificationError(f"non-finite JSON number: {value}")


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_manifest(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError("manifest is not strict UTF-8 JSON") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_decimal(value: str, *, allow_empty: bool = False) -> Decimal | None:
    if allow_empty and value == "":
        return None
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise VerificationError(f"invalid decimal: {value!r}") from exc
    if not result.is_finite():
        raise VerificationError(f"non-finite decimal: {value!r}")
    return result


def split_for(day: date) -> str:
    if day <= TRAIN_END:
        return "train"
    if day <= VALIDATION_END:
        return "validation"
    return "test"


def parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise VerificationError(f"invalid ISO settlement date: {value!r}") from exc


def source_url(name: str, month: str) -> str:
    if month in LEGACY_MONTHS:
        root = LEGACY_DOWNLOAD_ROOT
    elif month in OTHER_MONTHS:
        root = OTHER_DOWNLOAD_ROOT
    else:
        root = DOWNLOAD_ROOT
    return f"{root}/{name}"


def member_name_class(archive_name: str, member_name: str) -> str:
    expected_base = Path(archive_name).stem
    lowered = member_name.lower()
    if lowered == f"{expected_base}.txt".lower():
        return "expected_txt"
    if lowered == expected_base.lower():
        return "expected_extensionless"
    alias = KNOWN_MEMBER_ALIASES.get(archive_name)
    if alias is not None and lowered == alias.lower():
        return "known_exception"
    raise VerificationError(f"unexpected source member name in {archive_name}: {member_name!r}")


def read_source_trailer(path: Path) -> tuple[str, str, int, int]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) != 1:
                raise VerificationError(f"source member count mismatch: {path.name}")
            member_name = members[0].filename
            member_contract = member_name_class(path.name, member_name)
            with archive.open(members[0], "r") as handle:
                data = handle.read()
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise VerificationError(f"source ZIP cannot be read: {path.name}") from exc
    lines = [line.strip() for line in data.decode("cp1252").splitlines() if line.strip()]
    if len(lines) < 2:
        raise VerificationError(f"source trailer missing: {path.name}")
    record_match = TRAILER_RECORD_RE.fullmatch(lines[-2])
    quantity_match = TRAILER_QUANTITY_RE.fullmatch(lines[-1])
    if record_match is None or quantity_match is None:
        raise VerificationError(f"source trailer malformed: {path.name}")
    return (
        member_name,
        member_contract,
        int(record_match.group("value")),
        int(quantity_match.group("value")),
    )


def verify(stage_dir: Path, raw_dir: Path):
    stage_dir = stage_dir.resolve(strict=True)
    raw_dir = raw_dir.resolve(strict=True)
    if stage_dir.is_symlink() or raw_dir.is_symlink():
        raise VerificationError("stage and raw directories must not be symlinks")
    actual_files = {
        path.name for path in stage_dir.iterdir() if path.is_file() and not path.is_symlink()
    }
    if actual_files != EXPECTED_FILES or any(path.is_dir() for path in stage_dir.iterdir()):
        raise VerificationError(f"published member set mismatch: {sorted(actual_files)!r}")
    manifest = load_manifest(stage_dir / "manifest.json")
    if manifest.get("schema_version") != 2 or manifest.get("dataset") != "SEC Fails-to-Deliver":
        raise VerificationError("manifest identity mismatch")
    if set(manifest.get("interpretation", {})) != REQUIRED_WARNINGS:
        raise VerificationError("interpretation warning contract mismatch")
    if manifest.get("split", {}).get("unit") != "settlement_date":
        raise VerificationError("split unit must be settlement_date")
    boundaries = manifest.get("split", {}).get("boundaries", {})
    if boundaries != {
        "train_end_inclusive": "2023-12-31",
        "validation_end_inclusive": "2024-12-31",
    }:
        raise VerificationError("split boundaries mismatch")

    for name in ("fails.csv.gz", "daily_stress.csv"):
        metadata = manifest.get("files", {}).get(name)
        path = stage_dir / name
        if not isinstance(metadata, dict):
            raise VerificationError(f"missing file metadata: {name}")
        if metadata.get("bytes") != path.stat().st_size:
            raise VerificationError(f"file byte count mismatch: {name}")
        if metadata.get("sha256") != sha256_file(path):
            raise VerificationError(f"file SHA-256 mismatch: {name}")

    archives = manifest.get("source_archives")
    if not isinstance(archives, list) or len(archives) != manifest.get("archive_count"):
        raise VerificationError("source archive inventory mismatch")
    archive_by_name = {}
    previous_name = None
    for item in archives:
        name = item.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"cnsfails\d{6}[ab]\.zip", name):
            raise VerificationError(f"invalid source archive name: {name!r}")
        if name in archive_by_name or (previous_name is not None and name <= previous_name):
            raise VerificationError("source archive inventory is duplicate or unsorted")
        month = name[8:14]
        if item.get("url") != source_url(name, month):
            raise VerificationError(f"source URL mismatch: {name}")
        raw_path = raw_dir / name
        if raw_path.is_symlink() or not raw_path.is_file():
            raise VerificationError(f"source archive missing or non-regular: {name}")
        if item.get("bytes") != raw_path.stat().st_size:
            raise VerificationError(f"source archive byte count mismatch: {name}")
        if item.get("sha256") != sha256_file(raw_path):
            raise VerificationError(f"source archive SHA-256 mismatch: {name}")
        (
            member_name,
            member_contract,
            trailer_record_count,
            trailer_total_quantity,
        ) = read_source_trailer(raw_path)
        if item.get("member") != member_name:
            raise VerificationError(f"source member name mismatch: {name}")
        if item.get("member_name_contract") != member_contract:
            raise VerificationError(f"source member contract mismatch: {name}")
        if item.get("trailer_record_count") != trailer_record_count:
            raise VerificationError(f"source record trailer mismatch: {name}")
        if item.get("trailer_total_quantity_fails") != trailer_total_quantity:
            raise VerificationError(f"source quantity trailer mismatch: {name}")
        archive_by_name[name] = item
        previous_name = name

    daily = defaultdict(
        lambda: {
            "securities": 0,
            "shares": 0,
            "priced_shares": 0,
            "value": Decimal(0),
            "missing": 0,
        }
    )
    unique_cusips = set()
    row_count = 0
    split_rows = defaultdict(int)
    previous_day = None
    keys_for_day = set()
    archive_rows = defaultdict(int)
    archive_quantities = defaultdict(int)
    archive_quality = defaultdict(
        lambda: {
            "blank_symbol_rows": 0,
            "delimiter_repaired_rows": 0,
            "max_symbol_length": 0,
            "symbol_over_official_limit_rows": 0,
        }
    )
    source_quality = {
        "blank_symbol_rows": 0,
        "delimiter_repaired_rows": 0,
        "extensionless_member_archives": sum(
            item.get("member_name_contract") == "expected_extensionless" for item in archives
        ),
        "known_member_name_exception_archives": sum(
            item.get("member_name_contract") == "known_exception" for item in archives
        ),
        "max_symbol_length": 0,
        "symbol_over_official_limit_rows": 0,
    }
    total_quantity = 0
    with gzip.open(stage_dir / "fails.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise VerificationError("fails file is empty") from exc
        if header != FAILS_HEADER:
            raise VerificationError("fails header mismatch")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(FAILS_HEADER):
                raise VerificationError(f"fails row width mismatch at line {line_number}")
            day = parse_day(row[0])
            if previous_day is not None and day < previous_day:
                raise VerificationError("fails rows are not chronological")
            if previous_day != day:
                keys_for_day.clear()
                previous_day = day
            cusip = row[1]
            if not 1 <= len(cusip) <= 9 or cusip in keys_for_day:
                raise VerificationError(f"invalid or duplicate CUSIP at line {line_number}")
            keys_for_day.add(cusip)
            symbol = row[2]
            if (
                len(symbol) > SOURCE_SYMBOL_HARD_LIMIT
                or any(ord(character) < 32 or ord(character) == 127 for character in symbol)
                or not row[4]
                or len(row[4]) > 30
            ):
                raise VerificationError(f"SEC field limit violation at line {line_number}")
            if not row[3].isdigit() or int(row[3]) <= 0:
                raise VerificationError(f"invalid quantity at line {line_number}")
            quantity = int(row[3])
            price = exact_decimal(row[5], allow_empty=True)
            proxy = exact_decimal(row[6], allow_empty=True)
            if price is None:
                if proxy is not None:
                    raise VerificationError(f"proxy without price at line {line_number}")
            elif price <= 0 or proxy != price * quantity:
                raise VerificationError(f"price/proxy mismatch at line {line_number}")
            source = archive_by_name.get(row[7])
            if source is None or row[8] != source.get("half"):
                raise VerificationError(f"source reference mismatch at line {line_number}")
            if day.strftime("%Y%m") != source.get("month"):
                raise VerificationError(f"source month mismatch at line {line_number}")
            if row[8] == "a" and day.day > 14 or row[8] == "b" and day.day < 15:
                raise VerificationError(f"source half mismatch at line {line_number}")
            if row[9] != split_for(day):
                raise VerificationError(f"split mismatch at line {line_number}")
            stats = daily[day]
            stats["securities"] += 1
            stats["shares"] += quantity
            if price is None:
                stats["missing"] += 1
            else:
                stats["priced_shares"] += quantity
                stats["value"] += proxy
            unique_cusips.add(cusip)
            split_rows[row[9]] += 1
            archive_rows[row[7]] += 1
            archive_quantities[row[7]] += quantity
            total_quantity += quantity
            quality = archive_quality[row[7]]
            symbol_length = len(symbol)
            quality["max_symbol_length"] = max(quality["max_symbol_length"], symbol_length)
            source_quality["max_symbol_length"] = max(
                source_quality["max_symbol_length"], symbol_length
            )
            if not symbol:
                quality["blank_symbol_rows"] += 1
                source_quality["blank_symbol_rows"] += 1
            if symbol_length > OFFICIAL_MAX_SYMBOL_LENGTH:
                quality["symbol_over_official_limit_rows"] += 1
                source_quality["symbol_over_official_limit_rows"] += 1
            if "|" in row[4]:
                quality["delimiter_repaired_rows"] += 1
                source_quality["delimiter_repaired_rows"] += 1
            row_count += 1

    for name, source in archive_by_name.items():
        if archive_rows[name] != source.get("rows"):
            raise VerificationError(f"archive row reconciliation failed: {name}")
        if archive_rows[name] != source.get("trailer_record_count"):
            raise VerificationError(f"archive record trailer failed: {name}")
        if archive_quantities[name] != source.get("quantity_fails"):
            raise VerificationError(f"archive quantity reconciliation failed: {name}")
        if archive_quantities[name] != source.get("trailer_total_quantity_fails"):
            raise VerificationError(f"archive quantity trailer failed: {name}")
        if dict(archive_quality[name]) != source.get("field_quality"):
            raise VerificationError(f"archive field-quality mismatch: {name}")
    counts = manifest.get("counts", {})
    if counts != {
        "fails_rows": row_count,
        "quantity_fails": total_quantity,
        "settlement_dates": len(daily),
        "unique_cusips": len(unique_cusips),
    }:
        raise VerificationError("manifest count reconciliation failed")
    expected_source_quality = {
        **source_quality,
        "official_symbol_max_length": OFFICIAL_MAX_SYMBOL_LENGTH,
        "parser_symbol_hard_limit": SOURCE_SYMBOL_HARD_LIMIT,
        "trailer_reconciled_archives": len(archives),
    }
    if manifest.get("source_quality") != expected_source_quality:
        raise VerificationError("manifest source-quality reconciliation failed")
    if manifest.get("split", {}).get("row_counts") != dict(sorted(split_rows.items())):
        raise VerificationError("manifest split reconciliation failed")

    seen_days = []
    with (stage_dir / "daily_stress.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise VerificationError("daily file is empty") from exc
        if header != DAILY_HEADER:
            raise VerificationError("daily header mismatch")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(DAILY_HEADER):
                raise VerificationError(f"daily row width mismatch at line {line_number}")
            day = parse_day(row[0])
            if seen_days and day <= seen_days[-1]:
                raise VerificationError("daily rows are not strictly chronological")
            seen_days.append(day)
            stats = daily.get(day)
            if stats is None:
                raise VerificationError(f"daily row without fails rows: {day}")
            expected_share = Decimal(stats["missing"]) / Decimal(stats["securities"])
            actual = (
                int(row[1]),
                int(row[2]),
                int(row[3]),
                exact_decimal(row[4]),
                int(row[5]),
                exact_decimal(row[6]),
                row[7],
            )
            expected = (
                stats["securities"],
                stats["shares"],
                stats["priced_shares"],
                stats["value"],
                stats["missing"],
                expected_share.quantize(Decimal("0.000000000001")),
                split_for(day),
            )
            if actual != expected:
                raise VerificationError(f"daily reconciliation failed: {day}")
    if seen_days != sorted(daily):
        raise VerificationError("daily settlement-date coverage mismatch")

    result = {
        "archive_count": len(archives),
        "fails_rows": row_count,
        "file_integrity": "PASS",
        "interpretation_contract": "PASS",
        "quantity_fails": total_quantity,
        "settlement_dates": len(daily),
        "source_quality": expected_source_quality,
        "source_integrity": "PASS",
        "split_rows": dict(sorted(split_rows.items())),
        "unique_cusips": len(unique_cusips),
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify(args.stage_dir, args.raw_dir)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    print("SEC_FTD_STAGE_VERIFICATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
