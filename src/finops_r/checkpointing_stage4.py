from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .json_safety import require_json_object, strict_json_dumps, strict_json_loads

CHECKPOINT_SCHEMA_VERSION = "1.0"
_THREAD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CheckpointConflictError(RuntimeError):
    """Raised when replay inputs conflict with already persisted state."""


class ReleasedTraceError(RuntimeError):
    """Raised when a caller tries to mutate a released thread."""


@dataclass(frozen=True)
class Checkpoint:
    thread_id: str
    sequence: int
    kind: str
    payload: dict[str, Any]
    previous_sha256: str
    checkpoint_sha256: str


def _canonical(value: Any) -> str:
    return strict_json_dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checkpoint_sha256(
    *,
    thread_id: str,
    sequence: int,
    kind: str,
    payload_json: str,
    previous_sha256: str,
) -> str:
    return _sha256_text(
        _canonical(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "thread_id": thread_id,
                "sequence": sequence,
                "kind": kind,
                "payload": require_json_object(
                    strict_json_loads(payload_json),
                    source="checkpoint payload",
                ),
                "previous_sha256": previous_sha256,
            }
        )
    )


class SQLiteCheckpointStore:
    """Durable, hash-chained checkpoints plus an idempotent simulation ledger."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    input_sha256 TEXT NOT NULL,
                    released INTEGER NOT NULL DEFAULT 0 CHECK (released IN (0, 1)),
                    release_sha256 TEXT
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK (sequence >= 1),
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_sha256 TEXT NOT NULL,
                    checkpoint_sha256 TEXT NOT NULL,
                    PRIMARY KEY (thread_id, sequence),
                    UNIQUE (checkpoint_sha256),
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
                );
                CREATE TABLE IF NOT EXISTS simulation_effects (
                    idempotency_key TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL,
                    receipt_json TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _validate_thread_id(thread_id: str) -> None:
        if _THREAD_ID.fullmatch(thread_id) is None:
            raise ValueError("thread_id must be one safe canonical identifier")

    def bind_thread(self, thread_id: str, input_sha256: str) -> None:
        self._validate_thread_id(thread_id)
        if _SHA256.fullmatch(input_sha256) is None:
            raise ValueError("input_sha256 must be lowercase SHA-256")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT input_sha256 FROM threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO threads(thread_id, input_sha256) VALUES (?, ?)",
                    (thread_id, input_sha256),
                )
            elif row["input_sha256"] != input_sha256:
                raise CheckpointConflictError("thread is already bound to different inputs")
            connection.commit()

    def append(self, thread_id: str, kind: str, payload: dict[str, Any]) -> Checkpoint:
        self._validate_thread_id(thread_id)
        if not kind or kind != kind.strip() or len(kind) > 80:
            raise ValueError("checkpoint kind must be non-empty and trimmed")
        payload_json = _canonical(payload)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            thread = connection.execute(
                "SELECT released FROM threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if thread is None:
                raise CheckpointConflictError("thread must be bound before checkpointing")
            if thread["released"]:
                raise ReleasedTraceError("released checkpoint history is immutable")
            latest = connection.execute(
                """
                SELECT sequence, kind, payload_json, checkpoint_sha256
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
            if (
                latest is not None
                and latest["kind"] == kind
                and latest["payload_json"] == payload_json
            ):
                connection.commit()
                return self.load(thread_id)[-1]
            sequence = 1 if latest is None else int(latest["sequence"]) + 1
            previous = "0" * 64 if latest is None else str(latest["checkpoint_sha256"])
            checksum = _checkpoint_sha256(
                thread_id=thread_id,
                sequence=sequence,
                kind=kind,
                payload_json=payload_json,
                previous_sha256=previous,
            )
            connection.execute(
                """
                INSERT INTO checkpoints(
                    thread_id, sequence, kind, payload_json,
                    previous_sha256, checkpoint_sha256
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (thread_id, sequence, kind, payload_json, previous, checksum),
            )
            connection.commit()
        return Checkpoint(
            thread_id=thread_id,
            sequence=sequence,
            kind=kind,
            payload=payload,
            previous_sha256=previous,
            checkpoint_sha256=checksum,
        )

    def load(self, thread_id: str) -> tuple[Checkpoint, ...]:
        self._validate_thread_id(thread_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, kind, payload_json, previous_sha256, checkpoint_sha256
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY sequence
                """,
                (thread_id,),
            ).fetchall()
        checkpoints: list[Checkpoint] = []
        previous = "0" * 64
        for expected_sequence, row in enumerate(rows, start=1):
            sequence = int(row["sequence"])
            if sequence != expected_sequence or row["previous_sha256"] != previous:
                raise CheckpointConflictError("checkpoint chain sequence is invalid")
            payload = require_json_object(
                strict_json_loads(row["payload_json"]),
                source="checkpoint payload",
            )
            expected_sha = _checkpoint_sha256(
                thread_id=thread_id,
                sequence=sequence,
                kind=str(row["kind"]),
                payload_json=str(row["payload_json"]),
                previous_sha256=previous,
            )
            if row["checkpoint_sha256"] != expected_sha:
                raise CheckpointConflictError("checkpoint hash chain is invalid")
            checkpoints.append(
                Checkpoint(
                    thread_id=thread_id,
                    sequence=sequence,
                    kind=str(row["kind"]),
                    payload=payload,
                    previous_sha256=previous,
                    checkpoint_sha256=expected_sha,
                )
            )
            previous = expected_sha
        return tuple(checkpoints)

    def record_simulation_once(
        self,
        *,
        idempotency_key: str,
        request: dict[str, Any],
        receipt_factory: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        if _SHA256.fullmatch(idempotency_key) is None:
            raise ValueError("idempotency_key must be lowercase SHA-256")
        request_sha = _sha256_text(_canonical(request))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT request_sha256, receipt_json
                FROM simulation_effects
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                if row["request_sha256"] != request_sha:
                    raise CheckpointConflictError(
                        "idempotency key is already bound to a different simulation request"
                    )
                connection.commit()
                return require_json_object(
                    strict_json_loads(row["receipt_json"]),
                    source="simulation receipt",
                )
            receipt = receipt_factory()
            receipt_json = _canonical(receipt)
            connection.execute(
                """
                INSERT INTO simulation_effects(
                    idempotency_key, request_sha256, receipt_json
                ) VALUES (?, ?, ?)
                """,
                (idempotency_key, request_sha, receipt_json),
            )
            connection.commit()
            return receipt

    def simulation_effect_count(self, idempotency_key: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM simulation_effects WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return int(row["count"])

    def release(self, thread_id: str, state_sha256: str) -> dict[str, Any]:
        self._validate_thread_id(thread_id)
        if _SHA256.fullmatch(state_sha256) is None:
            raise ValueError("state_sha256 must be lowercase SHA-256")
        checkpoints = self.load(thread_id)
        if not checkpoints:
            raise CheckpointConflictError("cannot release an empty checkpoint history")
        release = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "thread_id": thread_id,
            "checkpoint_count": len(checkpoints),
            "last_checkpoint_sha256": checkpoints[-1].checkpoint_sha256,
            "state_sha256": state_sha256,
        }
        release_sha = _sha256_text(_canonical(release))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT released, release_sha256 FROM threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row is None:
                raise CheckpointConflictError("thread does not exist")
            if row["released"]:
                if row["release_sha256"] != release_sha:
                    raise CheckpointConflictError("released trace does not match requested state")
            else:
                connection.execute(
                    """
                    UPDATE threads
                    SET released = 1, release_sha256 = ?
                    WHERE thread_id = ?
                    """,
                    (release_sha, thread_id),
                )
            connection.commit()
        return {**release, "release_sha256": release_sha, "status": "RELEASED"}

    def verify_release(self, thread_id: str, state_sha256: str) -> dict[str, Any]:
        release = self.release(thread_id, state_sha256)
        checkpoints = self.load(thread_id)
        if release["last_checkpoint_sha256"] != checkpoints[-1].checkpoint_sha256:
            raise CheckpointConflictError("released trace head no longer matches checkpoint chain")
        return release
