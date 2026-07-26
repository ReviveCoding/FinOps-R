#!/usr/bin/env python3
"""Stage a fixed SEC fails-to-deliver snapshot into deterministic CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

DATASET_PAGE = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"
DOWNLOAD_ROOT = "https://www.sec.gov/files/data/fails-deliver-data"
LEGACY_DOWNLOAD_ROOT = "https://www.sec.gov/files/node/add/data_distribution"
OTHER_DOWNLOAD_ROOT = "https://www.sec.gov/files/data/other/fails-deliver-data"
EXPECTED_HEADER = (
    "SETTLEMENT DATE",
    "CUSIP",
    "SYMBOL",
    "QUANTITY (FAILS)",
    "DESCRIPTION",
    "PRICE",
)
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
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 250
OFFICIAL_MAX_SYMBOL_LENGTH = 10
SOURCE_SYMBOL_HARD_LIMIT = 32
ARCHIVE_RE = re.compile(r"^cnsfails(?P<month>\d{6})(?P<half>[ab])\.zip$")
TRAILER_RECORD_RE = re.compile(r"^Trailer record count (?P<value>\d+)$")
TRAILER_QUANTITY_RE = re.compile(r"^Trailer total quantity of shares (?P<value>\d+)$")
LEGACY_MONTHS = {"202002", "202003", "202004"}
OTHER_MONTHS = {"202605"}
KNOWN_MEMBER_ALIASES = {
    "cnsfails202204b.zip": "SEC Failed To Deliver April 2022 second half.txt",
    "cnsfails202308b.zip": "cnsfails202309a.txt",
}


class ContractError(RuntimeError):
    """Raised when source data violate the frozen staging contract."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    month: str
    half: str
    url: str


def parse_month(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"\d{6}", value):
        raise ContractError(f"invalid month: {value!r}")
    year, month = int(value[:4]), int(value[4:])
    if year < 2009 or not 1 <= month <= 12:
        raise ContractError(f"invalid month: {value!r}")
    return year, month


def iter_months(start: str, end: str):
    year, month = parse_month(start)
    end_year, end_month = parse_month(end)
    if (year, month) > (end_year, end_month):
        raise ContractError("start month is after end month")
    while (year, month) <= (end_year, end_month):
        yield f"{year:04d}{month:02d}"
        month += 1
        if month == 13:
            year += 1
            month = 1


def source_url(name: str, month: str) -> str:
    if not ARCHIVE_RE.fullmatch(name):
        raise ContractError(f"invalid archive name: {name!r}")
    if month in LEGACY_MONTHS:
        root = LEGACY_DOWNLOAD_ROOT
    elif month in OTHER_MONTHS:
        root = OTHER_DOWNLOAD_ROOT
    else:
        root = DOWNLOAD_ROOT
    return f"{root}/{name}"


def source_specs(start: str, end: str) -> list[SourceSpec]:
    return [
        SourceSpec(
            name=f"cnsfails{month}{half}.zip",
            month=month,
            half=half,
            url=source_url(f"cnsfails{month}{half}.zip", month),
        )
        for month in iter_months(start, end)
        for half in ("a", "b")
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for(day: date) -> str:
    if day <= TRAIN_END:
        return "train"
    if day <= VALIDATION_END:
        return "validation"
    return "test"


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def deterministic_gzip_text_writer(path: Path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return raw, compressed, text


def close_gzip_writer(raw, compressed, text) -> None:
    text.flush()
    text.detach()
    compressed.close()
    raw.close()


def decode_member(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise ContractError("archive member is neither UTF-8 nor Windows-1252 text")


def validate_zip_member(info: zipfile.ZipInfo) -> None:
    name = info.filename.replace("\\", "/")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
        raise ContractError(f"unsafe ZIP member path: {info.filename!r}")
    if info.is_dir():
        raise ContractError("directory member is not allowed")
    unix_mode = (info.external_attr >> 16) & 0o170000
    if unix_mode == 0o120000:
        raise ContractError("symbolic-link ZIP member is not allowed")
    if info.file_size <= 0 or info.file_size > MAX_MEMBER_BYTES:
        raise ContractError(f"invalid uncompressed member size: {info.file_size}")
    ratio = info.file_size / max(info.compress_size, 1)
    if ratio > MAX_COMPRESSION_RATIO:
        raise ContractError(f"ZIP compression ratio exceeds limit: {ratio:.2f}")


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
    raise ContractError(f"unexpected member name in {archive_name}: {member_name!r}")


def parse_day(value: str, spec: SourceSpec) -> date:
    if not re.fullmatch(r"\d{8}", value):
        raise ContractError(f"invalid settlement date: {value!r}")
    try:
        day = datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ContractError(f"invalid settlement date: {value!r}") from exc
    if day.strftime("%Y%m") != spec.month:
        raise ContractError(f"settlement date outside archive month: {value}")
    if spec.half == "a" and day.day > 14:
        raise ContractError(f"date outside first-half archive: {value}")
    if spec.half == "b" and day.day < 15:
        raise ContractError(f"date outside second-half archive: {value}")
    return day


def parse_price(value: str) -> Decimal | None:
    value = value.strip()
    if value in {"", "."}:
        return None
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ContractError(f"invalid price: {value!r}") from exc
    if not result.is_finite() or result <= 0:
        raise ContractError(f"invalid price: {value!r}")
    return result


def normalized_row(raw: list[str], spec: SourceSpec):
    if raw and raw[-1] == "" and len(raw) in {7, 8}:
        raw = raw[:-1]
    delimiter_repaired = False
    if len(raw) == 7 and not raw[5].strip():
        raw = raw[:4] + [raw[4] + "|" + raw[5]] + [raw[6]]
        delimiter_repaired = True
    if len(raw) != len(EXPECTED_HEADER):
        raise ContractError(f"expected 6 pipe-delimited fields, found {len(raw)}")
    day = parse_day(raw[0].strip(), spec)
    cusip = raw[1].strip()
    symbol = raw[2].strip()
    quantity_text = raw[3].strip()
    description = raw[4].strip()
    if not 1 <= len(cusip) <= 9 or any(ch.isspace() for ch in cusip):
        raise ContractError(f"invalid CUSIP: {cusip!r}")
    if len(symbol) > SOURCE_SYMBOL_HARD_LIMIT or any(
        ord(character) < 32 or ord(character) == 127 for character in symbol
    ):
        raise ContractError(f"invalid symbol: {symbol!r}")
    if not description or len(description) > 30:
        raise ContractError(f"invalid description length: {description!r}")
    if not quantity_text.isdigit() or int(quantity_text) <= 0:
        raise ContractError(f"invalid quantity: {quantity_text!r}")
    quantity = int(quantity_text)
    price = parse_price(raw[5])
    proxy = None if price is None else price * quantity
    return (
        day,
        cusip,
        symbol,
        quantity,
        description,
        price,
        proxy,
        delimiter_repaired,
    )


def load_archive(path: Path, spec: SourceSpec):
    size = path.stat().st_size
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise ContractError(f"archive size outside contract: {path.name}={size}")
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"source archive is not a regular file: {path}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) != 1:
                raise ContractError(f"archive must contain exactly one file: {path.name}")
            info = members[0]
            validate_zip_member(info)
            member_contract = member_name_class(path.name, info.filename)
            with archive.open(info, "r") as handle:
                data = handle.read(MAX_MEMBER_BYTES + 1)
            if len(data) != info.file_size or len(data) > MAX_MEMBER_BYTES:
                raise ContractError(f"member size mismatch: {path.name}")
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise ContractError(f"invalid ZIP archive: {path.name}") from exc
    text, encoding = decode_member(data)
    return info.filename, member_contract, encoding, text


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True)
        handle.write("\n")


def stage(raw_dir: Path, output_dir: Path, start_month: str, end_month: str):
    raw_dir = raw_dir.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists():
        raise ContractError(f"output directory already exists: {output_dir}")
    specs = source_specs(start_month, end_month)
    expected_names = {spec.name for spec in specs}
    missing = sorted(name for name in expected_names if not (raw_dir / name).is_file())
    if missing:
        raise ContractError(f"missing source archives: {', '.join(missing[:5])}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    archive_inventory = []
    daily = defaultdict(
        lambda: {
            "securities": 0,
            "shares": 0,
            "priced_shares": 0,
            "value": Decimal(0),
            "missing": 0,
        }
    )
    unique_cusips: set[str] = set()
    split_rows = defaultdict(int)
    row_count = 0
    total_quantity = 0
    source_quality = {
        "blank_symbol_rows": 0,
        "delimiter_repaired_rows": 0,
        "extensionless_member_archives": 0,
        "known_member_name_exception_archives": 0,
        "max_symbol_length": 0,
        "symbol_over_official_limit_rows": 0,
    }
    previous_day: date | None = None
    keys_for_day: set[str] = set()
    minimum_day: date | None = None
    maximum_day: date | None = None
    raw, compressed, text = deterministic_gzip_text_writer(temp_dir / "fails.csv.gz")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(FAILS_HEADER)
    try:
        for spec in specs:
            archive_path = raw_dir / spec.name
            member_name, member_contract, encoding, member_text = load_archive(archive_path, spec)
            if member_contract == "expected_extensionless":
                source_quality["extensionless_member_archives"] += 1
            elif member_contract == "known_exception":
                source_quality["known_member_name_exception_archives"] += 1
            reader = csv.reader(io.StringIO(member_text, newline=""), delimiter="|")
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ContractError(f"empty source member: {spec.name}") from exc
            if len(header) == 7 and header[-1] == "":
                header = header[:-1]
            header = [item.strip().lstrip("\ufeff") for item in header]
            if tuple(header) != EXPECTED_HEADER:
                raise ContractError(f"unexpected source header: {spec.name}: {header!r}")
            archive_rows = 0
            archive_quantity = 0
            archive_min = None
            archive_max = None
            trailer_record_count = None
            trailer_total_quantity = None
            trailer_started = False
            archive_quality = {
                "blank_symbol_rows": 0,
                "delimiter_repaired_rows": 0,
                "max_symbol_length": 0,
                "symbol_over_official_limit_rows": 0,
            }
            for line_number, source_row in enumerate(reader, start=2):
                if not source_row or all(not item.strip() for item in source_row):
                    continue
                if len(source_row) == 1:
                    trailer_text = source_row[0].strip()
                    record_match = TRAILER_RECORD_RE.fullmatch(trailer_text)
                    quantity_match = TRAILER_QUANTITY_RE.fullmatch(trailer_text)
                    if record_match:
                        if trailer_record_count is not None:
                            raise ContractError(
                                f"{spec.name}:{line_number}: duplicate record trailer"
                            )
                        trailer_record_count = int(record_match.group("value"))
                        trailer_started = True
                        continue
                    if quantity_match:
                        if trailer_total_quantity is not None:
                            raise ContractError(
                                f"{spec.name}:{line_number}: duplicate quantity trailer"
                            )
                        trailer_total_quantity = int(quantity_match.group("value"))
                        trailer_started = True
                        continue
                if trailer_started:
                    raise ContractError(
                        f"{spec.name}:{line_number}: data or invalid text after trailer"
                    )
                try:
                    (
                        day,
                        cusip,
                        symbol,
                        quantity,
                        description,
                        price,
                        proxy,
                        delimiter_repaired,
                    ) = normalized_row(source_row, spec)
                except ContractError as exc:
                    raise ContractError(f"{spec.name}:{line_number}: {exc}") from exc
                if previous_day is not None and day < previous_day:
                    raise ContractError(
                        f"settlement dates are not monotonic: {day} after {previous_day}"
                    )
                if previous_day != day:
                    keys_for_day.clear()
                    previous_day = day
                if cusip in keys_for_day:
                    raise ContractError(f"duplicate settlement_date/CUSIP: {day}/{cusip}")
                keys_for_day.add(cusip)
                split = split_for(day)
                writer.writerow(
                    (
                        day.isoformat(),
                        cusip,
                        symbol,
                        quantity,
                        description,
                        "" if price is None else decimal_text(price),
                        "" if proxy is None else decimal_text(proxy),
                        spec.name,
                        spec.half,
                        split,
                    )
                )
                stats = daily[day]
                stats["securities"] += 1
                stats["shares"] += quantity
                if price is None:
                    stats["missing"] += 1
                else:
                    stats["priced_shares"] += quantity
                    stats["value"] += proxy
                unique_cusips.add(cusip)
                split_rows[split] += 1
                row_count += 1
                total_quantity += quantity
                archive_rows += 1
                archive_quantity += quantity
                symbol_length = len(symbol)
                archive_quality["max_symbol_length"] = max(
                    archive_quality["max_symbol_length"], symbol_length
                )
                source_quality["max_symbol_length"] = max(
                    source_quality["max_symbol_length"], symbol_length
                )
                if not symbol:
                    archive_quality["blank_symbol_rows"] += 1
                    source_quality["blank_symbol_rows"] += 1
                if symbol_length > OFFICIAL_MAX_SYMBOL_LENGTH:
                    archive_quality["symbol_over_official_limit_rows"] += 1
                    source_quality["symbol_over_official_limit_rows"] += 1
                if delimiter_repaired:
                    archive_quality["delimiter_repaired_rows"] += 1
                    source_quality["delimiter_repaired_rows"] += 1
                archive_min = day if archive_min is None else min(archive_min, day)
                archive_max = day if archive_max is None else max(archive_max, day)
                minimum_day = day if minimum_day is None else min(minimum_day, day)
                maximum_day = day if maximum_day is None else max(maximum_day, day)
            if archive_rows == 0:
                raise ContractError(f"source archive contains no records: {spec.name}")
            if trailer_record_count is None or trailer_total_quantity is None:
                raise ContractError(f"source archive trailer is incomplete: {spec.name}")
            if trailer_record_count != archive_rows:
                raise ContractError(
                    f"record trailer mismatch: {spec.name}: "
                    f"trailer={trailer_record_count} parsed={archive_rows}"
                )
            if trailer_total_quantity != archive_quantity:
                raise ContractError(
                    f"quantity trailer mismatch: {spec.name}: "
                    f"trailer={trailer_total_quantity} parsed={archive_quantity}"
                )
            archive_inventory.append(
                {
                    "bytes": archive_path.stat().st_size,
                    "encoding": encoding,
                    "half": spec.half,
                    "max_settlement_date": archive_max.isoformat(),
                    "member": member_name,
                    "member_name_contract": member_contract,
                    "min_settlement_date": archive_min.isoformat(),
                    "month": spec.month,
                    "name": spec.name,
                    "field_quality": archive_quality,
                    "quantity_fails": archive_quantity,
                    "rows": archive_rows,
                    "sha256": sha256_file(archive_path),
                    "trailer_record_count": trailer_record_count,
                    "trailer_total_quantity_fails": trailer_total_quantity,
                    "url": spec.url,
                }
            )
        close_gzip_writer(raw, compressed, text)
        raw = compressed = text = None

        with (temp_dir / "daily_stress.csv").open("w", encoding="utf-8", newline="") as handle:
            daily_writer = csv.writer(handle, lineterminator="\n")
            daily_writer.writerow(DAILY_HEADER)
            for day in sorted(daily):
                stats = daily[day]
                missing_share = Decimal(stats["missing"]) / Decimal(stats["securities"])
                daily_writer.writerow(
                    (
                        day.isoformat(),
                        stats["securities"],
                        stats["shares"],
                        stats["priced_shares"],
                        decimal_text(stats["value"]),
                        stats["missing"],
                        decimal_text(missing_share.quantize(Decimal("0.000000000001"))),
                        split_for(day),
                    )
                )

        output_files = {}
        for name in ("fails.csv.gz", "daily_stress.csv"):
            path = temp_dir / name
            output_files[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest = {
            "archive_count": len(specs),
            "counts": {
                "fails_rows": row_count,
                "quantity_fails": total_quantity,
                "settlement_dates": len(daily),
                "unique_cusips": len(unique_cusips),
            },
            "dataset": "SEC Fails-to-Deliver",
            "dataset_page": DATASET_PAGE,
            "files": output_files,
            "interpretation": {
                "fail_value_proxy": "quantity_fails multiplied by SEC previous-day closing price when present",
                "missing_zero_semantics": "No row is not interpreted as zero outstanding fails for modeling joins without an explicit security universe.",
                "price_warning": "SEC does not guarantee that its price matches closing prices from other sources.",
                "short_sale_warning": "Fails-to-deliver are not necessarily caused by short selling and are not evidence of abusive or naked short selling.",
                "stock_warning": "quantity_fails is an aggregate net balance outstanding, not daily new fails; fail age cannot be inferred.",
                "use": "Independent market-settlement stress proxy; not a production label.",
            },
            "schema_version": 2,
            "source_archives": archive_inventory,
            "source_quality": {
                **source_quality,
                "official_symbol_max_length": OFFICIAL_MAX_SYMBOL_LENGTH,
                "parser_symbol_hard_limit": SOURCE_SYMBOL_HARD_LIMIT,
                "trailer_reconciled_archives": len(archive_inventory),
            },
            "source_period": {
                "actual_max_settlement_date": maximum_day.isoformat(),
                "actual_min_settlement_date": minimum_day.isoformat(),
                "end_month": end_month,
                "start_month": start_month,
            },
            "split": {
                "boundaries": {
                    "train_end_inclusive": TRAIN_END.isoformat(),
                    "validation_end_inclusive": VALIDATION_END.isoformat(),
                },
                "ordering": "settlement_date",
                "row_counts": dict(sorted(split_rows.items())),
                "unit": "settlement_date",
            },
        }
        write_json(temp_dir / "manifest.json", manifest)
        os.replace(temp_dir, output_dir)
        return manifest
    except Exception:
        if text is not None:
            try:
                text.close()
            except Exception:
                pass
        if compressed is not None:
            try:
                compressed.close()
            except Exception:
                pass
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-month", default="202001")
    parser.add_argument("--end-month", default="202606")
    args = parser.parse_args(argv)
    manifest = stage(args.raw_dir, args.output_dir, args.start_month, args.end_month)
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    print("SEC_FTD_STAGING=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
