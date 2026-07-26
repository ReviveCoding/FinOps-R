from __future__ import annotations

import errno
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, IO


class StrictJSONError(ValueError):
    """Raised when JSON violates the project's strict interchange contract."""


_IS_WINDOWS = os.name == "nt"
_WINDOWS_READ_RETRY_SECONDS = 5.0
_WINDOWS_READ_MAX_DELAY_SECONDS = 0.05
_TRANSIENT_WINDOWS_READ_ERRORS = {5, 32, 33}


def _is_transient_windows_read_error(exc: PermissionError) -> bool:
    """Recognize native or POSIX-translated Windows sharing failures."""

    winerror = getattr(exc, "winerror", None)
    return winerror in _TRANSIENT_WINDOWS_READ_ERRORS or exc.errno == errno.EACCES


def _read_utf8_path(path: Path) -> str:
    """Read a path, tolerating transient Windows replace/share conflicts."""

    deadline = time.monotonic() + _WINDOWS_READ_RETRY_SECONDS
    delay = 0.001
    while True:
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError as exc:
            if (
                not _IS_WINDOWS
                or not _is_transient_windows_read_error(exc)
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(delay)
            delay = min(delay * 2, _WINDOWS_READ_MAX_DELAY_SECONDS)


def _reject_constant(token: str) -> Any:
    raise StrictJSONError(f"non-finite JSON number is prohibited: {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise StrictJSONError(f"JSON number is outside the finite float range: {token}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key is prohibited: {key!r}")
        result[key] = value
    return result


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Decode one JSON document and reject extensions or ambiguous objects."""

    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
        )
    except StrictJSONError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrictJSONError(str(exc)) from exc


def strict_json_load(source: Path | str | IO[str]) -> Any:
    """Decode strict JSON from a UTF-8 path or an open text stream."""

    if hasattr(source, "read"):
        return strict_json_loads(source.read())
    return strict_json_loads(_read_utf8_path(Path(source)))


def strict_json_dumps(value: Any, **kwargs: Any) -> str:
    """Encode standards-compliant JSON and reject non-finite numbers."""

    if kwargs.get("allow_nan") is True:
        raise TypeError("strict_json_dumps does not permit allow_nan=True")
    options = dict(kwargs)
    options["allow_nan"] = False
    try:
        return json.dumps(value, **options)
    except (TypeError, ValueError) as exc:
        raise StrictJSONError(str(exc)) from exc


def strict_json_dump(value: Any, handle: IO[str], **kwargs: Any) -> None:
    """Write standards-compliant JSON to an open text stream."""

    handle.write(strict_json_dumps(value, **kwargs))


def require_json_object(value: Any, *, source: str = "JSON document") -> dict[str, Any]:
    """Require a decoded top-level JSON object with string keys."""

    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise StrictJSONError(f"{source} must contain a top-level JSON object")
    return dict(value)
