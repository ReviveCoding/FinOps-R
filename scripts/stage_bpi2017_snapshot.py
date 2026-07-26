from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_ARCHIVE_BYTES = 29_658_747
EXPECTED_ARCHIVE_MD5 = "10b37a2f78e870d78406198403ff13d2"
EXPECTED_APPLICATIONS = 31_509
EXPECTED_EVENTS = 1_202_267
EXPECTED_OFFERS = 42_995
EXPECTED_ORIGINATORS = 149
MAX_UNCOMPRESSED_BYTES = 3_000_000_000
ATTRIBUTE_TAGS = {"string", "date", "int", "float", "boolean", "id"}


class BpiContractError(ValueError):
    """Raised when the BPI 2017 snapshot violates the staging contract."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _json(value: dict[str, str]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(path: Path, algorithm: str = "sha256") -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise BpiContractError(f"expected one regular file: {path}")
    digest = hashlib.new(algorithm)
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _parse_timestamp(value: str, context: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise BpiContractError(f"{context}: invalid ISO-8601 timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BpiContractError(f"{context}: timestamp must include an offset")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _attributes(element: ET.Element, context: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for child in element:
        if _local(child.tag) not in ATTRIBUTE_TAGS:
            continue
        key = child.attrib.get("key")
        value = child.attrib.get("value")
        if not key or value is None:
            raise BpiContractError(f"{context}: XES attribute must have key and value")
        if key in result:
            raise BpiContractError(f"{context}: duplicate XES attribute {key!r}")
        if "\x00" in key or "\x00" in value:
            raise BpiContractError(f"{context}: NUL bytes are forbidden")
        result[key] = value
    return result


def _required(attrs: dict[str, str], key: str, context: str) -> str:
    value = attrs.get(key, "")
    if not value.strip():
        raise BpiContractError(f"{context}: required attribute {key!r} is missing")
    return value


def _pseudonym(resource: str, archive_md5: str) -> str:
    digest = hashlib.sha256(f"bpi2017:{archive_md5}:{resource}".encode()).hexdigest()
    return f"resource_{digest[:20]}"


def _safe_attributes(attrs: dict[str, str], archive_md5: str) -> dict[str, str]:
    safe = dict(attrs)
    if safe.get("org:resource"):
        safe["org:resource"] = _pseudonym(safe["org:resource"], archive_md5)
    return safe


def _split_boundaries(starts: Counter[str], total: int) -> tuple[str, str]:
    if len(starts) < 3:
        raise BpiContractError("at least three distinct case start timestamps are required")
    train_target = max(1, int(total * 0.60))
    validation_target = min(total - 1, max(train_target + 1, int(total * 0.80)))
    cumulative = 0
    train_end: str | None = None
    validation_end: str | None = None
    for timestamp in sorted(starts):
        cumulative += starts[timestamp]
        if train_end is None and cumulative >= train_target:
            train_end = timestamp
        if validation_end is None and cumulative >= validation_target:
            validation_end = timestamp
            break
    if train_end is None or validation_end is None or train_end >= validation_end:
        raise BpiContractError("could not construct non-overlapping chronological splits")
    return train_end, validation_end


def _split(start: str, train_end: str, validation_end: str) -> str:
    if start <= train_end:
        return "train"
    if start <= validation_end:
        return "validation"
    return "test"


def _iter_traces(source: Path) -> Iterable[ET.Element]:
    uncompressed = 0
    try:
        with gzip.open(source, "rb") as compressed:
            parser = ET.XMLPullParser(events=("end",))
            while True:
                block = compressed.read(1024 * 1024)
                if not block:
                    break
                uncompressed += len(block)
                if uncompressed > MAX_UNCOMPRESSED_BYTES:
                    raise BpiContractError(
                        f"uncompressed XES exceeds {MAX_UNCOMPRESSED_BYTES} bytes"
                    )
                parser.feed(block)
                for _, element in parser.read_events():
                    if _local(element.tag) == "trace":
                        yield element
                        element.clear()
            parser.close()
    except (gzip.BadGzipFile, EOFError, ET.ParseError, OSError) as exc:
        raise BpiContractError(f"cannot stream valid gzip/XES from {source}") from exc


def stage_bpi2017_snapshot(
    source: Path,
    output: Path,
    *,
    expected_bytes: int = EXPECTED_ARCHIVE_BYTES,
    expected_md5: str = EXPECTED_ARCHIVE_MD5,
    expected_applications: int = EXPECTED_APPLICATIONS,
    expected_events: int = EXPECTED_EVENTS,
    expected_offers: int = EXPECTED_OFFERS,
    expected_originators: int = EXPECTED_ORIGINATORS,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset directory: {output}")
    for name, value in {
        "expected_bytes": expected_bytes,
        "expected_applications": expected_applications,
        "expected_events": expected_events,
        "expected_offers": expected_offers,
        "expected_originators": expected_originators,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    expected_md5 = expected_md5.lower()
    if len(expected_md5) != 32 or any(c not in "0123456789abcdef" for c in expected_md5):
        raise ValueError("expected_md5 must be 32 hexadecimal characters")

    archive_bytes, archive_md5 = _digest(source, "md5")
    _, archive_sha256 = _digest(source)
    if archive_bytes != expected_bytes:
        raise BpiContractError(
            f"archive byte mismatch: expected {expected_bytes}, got {archive_bytes}"
        )
    if archive_md5 != expected_md5:
        raise BpiContractError(f"archive MD5 mismatch: expected {expected_md5}, got {archive_md5}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    applications_base = staging / ".applications_base.csv"
    events_path = staging / "events.csv.gz"
    offers_path = staging / "offers.csv"
    starts: Counter[str] = Counter()
    activity_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    application_ids: set[str] = set()
    offer_ids: set[str] = set()
    originators: set[str] = set()
    offer_rows: dict[str, list[Any]] = {}
    event_count = 0

    try:
        with (
            applications_base.open("x", encoding="utf-8", newline="") as app_stream,
            gzip.open(events_path, "xt", encoding="utf-8", newline="") as event_stream,
        ):
            app_writer = csv.writer(app_stream, lineterminator="\n")
            event_writer = csv.writer(event_stream, lineterminator="\n")
            app_writer.writerow(
                [
                    "application_id",
                    "case_start_utc",
                    "case_end_utc",
                    "event_count",
                    "offer_count",
                    "terminal_activity",
                    "application_attributes_json",
                ]
            )
            event_writer.writerow(
                [
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
            )

            for case_number, trace in enumerate(_iter_traces(source), start=1):
                trace_attrs = _attributes(trace, f"trace {case_number}")
                application_id = _required(trace_attrs, "concept:name", f"trace {case_number}")
                if application_id != application_id.strip():
                    raise BpiContractError(f"trace {case_number}: application ID is not trimmed")
                if application_id in application_ids:
                    raise BpiContractError(f"duplicate application ID {application_id!r}")
                application_ids.add(application_id)

                case_events: list[tuple[dict[str, str], datetime]] = []
                previous: datetime | None = None
                case_offers: set[str] = set()
                for event_number, event in enumerate(
                    (child for child in trace if _local(child.tag) == "event"), start=1
                ):
                    context = f"application {application_id!r} event {event_number}"
                    attrs = _attributes(event, context)
                    activity = _required(attrs, "concept:name", context)
                    timestamp = _parse_timestamp(
                        _required(attrs, "time:timestamp", context), context
                    )
                    if previous is not None and timestamp < previous:
                        raise BpiContractError(f"{context}: timestamps are not nondecreasing")
                    previous = timestamp
                    case_events.append((attrs, timestamp))
                    activity_counts[activity] += 1
                    origin_counts[attrs.get("EventOrigin", "<missing>")] += 1
                    resource = attrs.get("org:resource", "")
                    if resource:
                        originators.add(resource)
                    offer_id = attrs.get("OfferID", "")
                    if offer_id:
                        case_offers.add(offer_id)
                        if offer_id in offer_rows and offer_rows[offer_id][0] != application_id:
                            raise BpiContractError(
                                f"offer {offer_id!r} is linked to more than one application"
                            )
                        if offer_id not in offer_rows:
                            offer_rows[offer_id] = [
                                application_id,
                                _iso(timestamp),
                                _iso(timestamp),
                                1,
                                activity,
                            ]
                        else:
                            row = offer_rows[offer_id]
                            row[2] = _iso(timestamp)
                            row[3] += 1
                            row[4] = activity

                    safe = _safe_attributes(attrs, archive_md5)
                    event_writer.writerow(
                        [
                            application_id,
                            event_number,
                            _iso(timestamp),
                            activity,
                            attrs.get("EventOrigin", ""),
                            attrs.get("lifecycle:transition", ""),
                            offer_id,
                            _pseudonym(resource, archive_md5) if resource else "",
                            _json(safe),
                        ]
                    )
                    event_count += 1

                if not case_events:
                    raise BpiContractError(f"application {application_id!r} has no events")
                start = _iso(case_events[0][1])
                end = _iso(case_events[-1][1])
                starts[start] += 1
                offer_ids.update(case_offers)
                app_writer.writerow(
                    [
                        application_id,
                        start,
                        end,
                        len(case_events),
                        len(case_offers),
                        case_events[-1][0]["concept:name"],
                        _json(trace_attrs),
                    ]
                )

        if len(application_ids) != expected_applications:
            raise BpiContractError(
                f"application count mismatch: expected {expected_applications}, "
                f"got {len(application_ids)}"
            )
        if event_count != expected_events:
            raise BpiContractError(
                f"event count mismatch: expected {expected_events}, got {event_count}"
            )
        if len(offer_ids) != expected_offers:
            raise BpiContractError(
                f"offer count mismatch: expected {expected_offers}, got {len(offer_ids)}"
            )
        if len(originators) != expected_originators:
            raise BpiContractError(
                f"originator count mismatch: expected {expected_originators}, "
                f"got {len(originators)}"
            )

        train_end, validation_end = _split_boundaries(starts, len(application_ids))
        split_counts: Counter[str] = Counter()
        applications_path = staging / "applications.csv"
        with (
            applications_base.open("r", encoding="utf-8", newline="") as reader_stream,
            applications_path.open("x", encoding="utf-8", newline="") as writer_stream,
        ):
            reader = csv.DictReader(reader_stream)
            fieldnames = list(reader.fieldnames or []) + ["split"]
            writer = csv.DictWriter(writer_stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                row["split"] = _split(row["case_start_utc"], train_end, validation_end)
                split_counts[row["split"]] += 1
                writer.writerow(row)
        applications_base.unlink()

        with offers_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                [
                    "application_id",
                    "offer_id",
                    "first_timestamp_utc",
                    "last_timestamp_utc",
                    "event_count",
                    "terminal_activity",
                ]
            )
            for offer_id in sorted(offer_rows):
                writer.writerow([offer_rows[offer_id][0], offer_id, *offer_rows[offer_id][1:]])

        if any(split_counts[key] < 1 for key in ("train", "validation", "test")):
            raise BpiContractError("chronological split produced an empty partition")

        file_manifest: dict[str, dict[str, Any]] = {}
        for path in (applications_path, events_path, offers_path):
            size, sha256 = _digest(path)
            file_manifest[path.name] = {"bytes": size, "sha256": sha256}
        manifest: dict[str, Any] = {
            "dataset": "BPI Challenge 2017",
            "doi": "10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b",
            "source_archive": {
                "name": source.name,
                "bytes": archive_bytes,
                "md5": archive_md5,
                "sha256": archive_sha256,
            },
            "counts": {
                "applications": len(application_ids),
                "events": event_count,
                "offers": len(offer_ids),
                "originators": len(originators),
            },
            "split": {
                "unit": "application",
                "ordering": "first_event_timestamp_utc",
                "train_end_inclusive": train_end,
                "validation_end_inclusive": validation_end,
                "counts": {key: split_counts[key] for key in ("train", "validation", "test")},
            },
            "privacy": {
                "raw_org_resource_published": False,
                "resource_transform": "sha256-derived deterministic pseudonym",
                "warning": "Pseudonymization is not a claim of irreversible anonymization.",
            },
            "activity_counts": dict(sorted(activity_counts.items())),
            "event_origin_counts": dict(sorted(origin_counts.items())),
            "files": file_manifest,
        }
        _atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage the exact BPI Challenge 2017 XES snapshot")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-bytes", type=int, default=EXPECTED_ARCHIVE_BYTES)
    parser.add_argument("--expected-md5", default=EXPECTED_ARCHIVE_MD5)
    parser.add_argument("--expected-applications", type=int, default=EXPECTED_APPLICATIONS)
    parser.add_argument("--expected-events", type=int, default=EXPECTED_EVENTS)
    parser.add_argument("--expected-offers", type=int, default=EXPECTED_OFFERS)
    parser.add_argument("--expected-originators", type=int, default=EXPECTED_ORIGINATORS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        manifest = stage_bpi2017_snapshot(
            args.source,
            args.output,
            expected_bytes=args.expected_bytes,
            expected_md5=args.expected_md5,
            expected_applications=args.expected_applications,
            expected_events=args.expected_events,
            expected_offers=args.expected_offers,
            expected_originators=args.expected_originators,
        )
    except Exception as exc:
        print(f"BPI2017_STAGING_ERROR={type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    print("BPI2017_STAGING=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
