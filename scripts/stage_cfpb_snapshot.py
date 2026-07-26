from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


CFPB_COLUMNS = (
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Consumer consent provided?",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Consumer disputed?",
    "Complaint ID",
)
CRITICAL_NONEMPTY = (
    "Date received",
    "Product",
    "Issue",
    "Consumer complaint narrative",
    "Company",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Complaint ID",
)
DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y")
MAX_SOURCE_BYTES = 4_000_000_000


class CfpbContractError(ValueError):
    """Raised when a CFPB snapshot violates the declared staging contract."""


def _sha256_copy(source: Path, destination: Path, max_bytes: int) -> tuple[int, str]:
    if source.is_symlink() or not source.is_file():
        raise CfpbContractError(f"source must be one regular file: {source}")
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            copied += len(block)
            if copied > max_bytes:
                raise CfpbContractError(f"source exceeds {max_bytes} bytes")
            digest.update(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    return copied, digest.hexdigest()


def _parse_date(value: str, field: str, row_number: int) -> tuple[str, str]:
    for date_format in DATE_FORMATS:
        try:
            parsed = datetime.strptime(value, date_format).date()
            return parsed.isoformat(), date_format
        except ValueError:
            continue
    raise CfpbContractError(f"row {row_number}: {field} must use MM/DD/YY or MM/DD/YYYY")


def _split_boundaries(date_counts: Counter[str], total: int) -> tuple[str, str]:
    if len(date_counts) < 3:
        raise CfpbContractError("at least three distinct received dates are required")
    train_target = total * 0.60
    validation_target = total * 0.80
    cumulative = 0
    train_end: str | None = None
    validation_end: str | None = None
    for date in sorted(date_counts):
        cumulative += date_counts[date]
        if train_end is None and cumulative >= train_target:
            train_end = date
        if validation_end is None and cumulative >= validation_target:
            validation_end = date
            break
    if train_end is None or validation_end is None or train_end >= validation_end:
        raise CfpbContractError("could not construct non-overlapping chronological splits")
    return train_end, validation_end


def _counter(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _load_repair_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CfpbContractError(f"cannot read repair report: {path}") from exc
    required = {
        "repair_type",
        "original_archive_bytes",
        "original_archive_sha256",
        "original_csv_bytes",
        "original_csv_sha256",
        "clean_csv_bytes",
        "clean_csv_sha256",
        "quarantined_bytes",
        "quarantined_sha256",
    }
    if not isinstance(report, dict) or set(report) != required:
        raise CfpbContractError("repair report has an incompatible schema")
    if report["repair_type"] != "remove_exact_truncated_final_record":
        raise CfpbContractError("unsupported repair type")
    for key in (
        "original_archive_bytes",
        "original_csv_bytes",
        "clean_csv_bytes",
        "quarantined_bytes",
    ):
        if isinstance(report[key], bool) or not isinstance(report[key], int) or report[key] < 0:
            raise CfpbContractError(f"repair report {key} must be a non-negative integer")
    for key in (
        "original_archive_sha256",
        "original_csv_sha256",
        "clean_csv_sha256",
        "quarantined_sha256",
    ):
        value = report[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise CfpbContractError(f"repair report {key} must be lowercase SHA-256")
    return report


def stage_cfpb_snapshot(
    source: Path,
    output: Path,
    *,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    min_rows: int = 1_000,
    source_archive_name: str | None = None,
    repair_report_path: Path | None = None,
    quarantine_path: Path | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset directory: {output}")
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 1:
        raise ValueError("min_rows must be a positive integer")
    if expected_rows is not None and (
        isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows < 1
    ):
        raise ValueError("expected_rows must be a positive integer")
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.lower()
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise ValueError("expected_sha256 must be 64 hexadecimal characters")

    repair_report = _load_repair_report(repair_report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        snapshot = staging / "complaints.csv"
        source_bytes, source_sha256 = _sha256_copy(source, snapshot, MAX_SOURCE_BYTES)
        if expected_sha256 is not None and source_sha256 != expected_sha256:
            raise CfpbContractError(
                f"source SHA-256 mismatch: expected {expected_sha256}, got {source_sha256}"
            )
        if repair_report is not None:
            if repair_report["clean_csv_bytes"] != source_bytes:
                raise CfpbContractError("repair report clean byte count does not match source")
            if repair_report["clean_csv_sha256"] != source_sha256:
                raise CfpbContractError("repair report clean SHA-256 does not match source")

        csv.field_size_limit(10_000_000)
        ids: set[str] = set()
        date_counts: Counter[str] = Counter()
        date_formats: Counter[str] = Counter()
        sent_date_formats: Counter[str] = Counter()
        products: Counter[str] = Counter()
        issues: Counter[str] = Counter()
        responses: Counter[str] = Counter()
        timely: Counter[str] = Counter()
        consent: Counter[str] = Counter()
        narrative_characters = 0
        narrative_min: int | None = None
        narrative_max = 0
        row_count = 0
        id_dates = staging / ".id_dates.csv"

        try:
            source_stream = snapshot.open("r", encoding="utf-8-sig", errors="strict", newline="")
        except OSError as exc:
            raise CfpbContractError(f"cannot read source snapshot: {snapshot}") from exc
        with source_stream, id_dates.open("x", encoding="utf-8", newline="") as pair_stream:
            reader = csv.reader(source_stream, strict=True)
            pair_writer = csv.writer(pair_stream, lineterminator="\n")
            try:
                header = next(reader)
            except StopIteration as exc:
                raise CfpbContractError("source CSV is empty") from exc
            if tuple(header) != CFPB_COLUMNS:
                raise CfpbContractError(
                    f"CFPB schema mismatch; expected {list(CFPB_COLUMNS)}, got {header}"
                )
            positions = {name: index for index, name in enumerate(header)}
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(CFPB_COLUMNS):
                    raise CfpbContractError(
                        f"row {row_number}: expected {len(CFPB_COLUMNS)} fields, got {len(row)}"
                    )
                if any("\x00" in value for value in row):
                    raise CfpbContractError(f"row {row_number}: NUL bytes are not allowed")
                for field in CRITICAL_NONEMPTY:
                    value = row[positions[field]]
                    if not value.strip():
                        raise CfpbContractError(f"row {row_number}: {field} must be non-empty")

                complaint_id = row[positions["Complaint ID"]]
                if (
                    complaint_id != complaint_id.strip()
                    or not complaint_id.isascii()
                    or not complaint_id.isdigit()
                ):
                    raise CfpbContractError(
                        f"row {row_number}: Complaint ID must be trimmed ASCII digits"
                    )
                if complaint_id in ids:
                    raise CfpbContractError(
                        f"row {row_number}: duplicate Complaint ID {complaint_id}"
                    )
                ids.add(complaint_id)

                received, received_format = _parse_date(
                    row[positions["Date received"]], "Date received", row_number
                )
                sent, sent_format = _parse_date(
                    row[positions["Date sent to company"]], "Date sent to company", row_number
                )
                if sent < received:
                    raise CfpbContractError(
                        f"row {row_number}: Date sent to company precedes Date received"
                    )
                response_value = row[positions["Timely response?"]]
                if response_value not in {"Yes", "No"}:
                    raise CfpbContractError(f"row {row_number}: Timely response? must be Yes or No")

                narrative_length = len(row[positions["Consumer complaint narrative"]])
                narrative_characters += narrative_length
                narrative_min = (
                    narrative_length
                    if narrative_min is None
                    else min(narrative_min, narrative_length)
                )
                narrative_max = max(narrative_max, narrative_length)
                date_counts[received] += 1
                date_formats[received_format] += 1
                sent_date_formats[sent_format] += 1
                products[row[positions["Product"]]] += 1
                issues[row[positions["Issue"]]] += 1
                responses[row[positions["Company response to consumer"]]] += 1
                timely[response_value] += 1
                consent[row[positions["Consumer consent provided?"]]] += 1
                pair_writer.writerow((complaint_id, received))
                row_count += 1

        if row_count < min_rows:
            raise CfpbContractError(f"snapshot has {row_count} rows; minimum is {min_rows}")
        if expected_rows is not None and row_count != expected_rows:
            raise CfpbContractError(
                f"row count mismatch: expected {expected_rows}, got {row_count}"
            )

        train_end, validation_end = _split_boundaries(date_counts, row_count)
        split_counts: Counter[str] = Counter()
        split_path = staging / "temporal_splits.csv"
        with (
            id_dates.open("r", encoding="utf-8", newline="") as source_pairs,
            split_path.open("x", encoding="utf-8", newline="") as split_stream,
        ):
            pair_reader = csv.reader(source_pairs, strict=True)
            split_writer = csv.writer(split_stream, lineterminator="\n")
            split_writer.writerow(("Complaint ID", "Date received", "split"))
            for complaint_id, received in pair_reader:
                if received <= train_end:
                    split = "train"
                elif received <= validation_end:
                    split = "validation"
                else:
                    split = "test"
                split_counts[split] += 1
                split_writer.writerow((complaint_id, received, split))
        id_dates.unlink()
        if set(split_counts) != {"train", "validation", "test"}:
            raise CfpbContractError("chronological split produced an empty partition")

        date_range = {"start": min(date_counts), "end": max(date_counts)}
        taxonomy = {
            "schema_version": "1.0",
            "product_counts": _counter(products),
            "issue_counts": _counter(issues),
            "company_response_counts": _counter(responses),
            "timely_response_counts": _counter(timely),
            "consumer_consent_counts": _counter(consent),
        }
        (staging / "label_taxonomy.json").write_text(
            json.dumps(taxonomy, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        if repair_report is not None:
            (staging / "source_repair.json").write_text(
                json.dumps(repair_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if quarantine_path is None or not quarantine_path.is_file():
                raise CfpbContractError("repair report requires a quarantine file")
            quarantine_bytes = quarantine_path.read_bytes()
            quarantine_sha256 = hashlib.sha256(quarantine_bytes).hexdigest()
            if len(quarantine_bytes) != repair_report["quarantined_bytes"]:
                raise CfpbContractError("quarantine byte count does not match repair report")
            if quarantine_sha256 != repair_report["quarantined_sha256"]:
                raise CfpbContractError("quarantine SHA-256 does not match repair report")
            quarantine_dir = staging / "quarantine"
            quarantine_dir.mkdir()
            (quarantine_dir / "truncated_final_record.bin").write_bytes(quarantine_bytes)

        manifest = {
            "schema_version": "1.0",
            "dataset_id": "cfpb_consumer_complaints",
            "classification": "public rolling proxy snapshot with opt-in complaint narratives",
            "official_source": "https://www.consumerfinance.gov/data-research/consumer-complaints/",
            "source_archive_name": source_archive_name,
            "source_snapshot": "complaints.csv",
            "source_bytes": source_bytes,
            "source_sha256": source_sha256,
            "row_count": row_count,
            "unique_complaint_ids": len(ids),
            "column_count": len(CFPB_COLUMNS),
            "columns": list(CFPB_COLUMNS),
            "date_range": date_range,
            "date_format_counts": _counter(date_formats),
            "sent_date_format_counts": _counter(sent_date_formats),
            "temporal_split_boundaries": {
                "train_end_inclusive": train_end,
                "validation_end_inclusive": validation_end,
            },
            "temporal_split_counts": _counter(split_counts),
            "narrative_statistics": {
                "count": row_count,
                "characters_total": narrative_characters,
                "characters_mean": narrative_characters / row_count,
                "characters_min": narrative_min,
                "characters_max": narrative_max,
            },
            "repair": repair_report,
            "use": "separate financial narrative triage and retrieval benchmark",
            "joined_to_tradeops_case_labels": False,
            "claim_boundary": (
                "CFPB complaints are opt-in, unverified narratives and are not a statistical "
                "sample of consumers or evidence of bank-production performance. This snapshot "
                "must remain separate from TradeOps-Sim case labels."
            ),
        }
        (staging / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            "# CFPB consumer-complaint narrative snapshot\n\n"
            "This directory was generated by `scripts/stage_cfpb_snapshot.py`. The data are a "
            "frozen public proxy for narrative triage and retrieval. They are not joined to "
            "TradeOps-Sim labels and are not representative of all consumers or institutions. "
            "See `dataset_manifest.json` and `source_repair.json` for provenance and repair "
            "evidence.\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and atomically stage a CFPB CSV snapshot"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--min-rows", type=int, default=1_000)
    parser.add_argument("--source-archive-name")
    parser.add_argument("--repair-report", type=Path)
    parser.add_argument("--quarantine", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = stage_cfpb_snapshot(
            args.source,
            args.output,
            expected_sha256=args.expected_sha256,
            expected_rows=args.expected_rows,
            min_rows=args.min_rows,
            source_archive_name=args.source_archive_name,
            repair_report_path=args.repair_report,
            quarantine_path=args.quarantine,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print(json.dumps({"status": "ERROR", "type": "KeyboardInterrupt"}), file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            json.dumps({"status": "ERROR", "type": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
