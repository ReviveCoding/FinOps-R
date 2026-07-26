from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .artifacts import atomic_json, sha256_file
from .constants import ACTION_BY_CAUSE
from .json_safety import require_json_object, strict_json_dumps, strict_json_load


SCHEMA_VERSION = "1.0"
CARD_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,95}$")
VALID_ACTIONS = frozenset(ACTION_BY_CAUSE.values()) | {"HOLD_FOR_REVIEW"}


def _canonical_payload(payload: dict[str, Any]) -> bytes:
    return strict_json_dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _card_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(payload)).hexdigest()


class PolicyCardSource(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    card_id: str
    title: str = Field(min_length=3, max_length=200)
    source_title: str = Field(min_length=3, max_length=300)
    source_url: str = Field(min_length=12, max_length=1000)
    source_date: date
    effective_from: datetime
    effective_to: datetime | None = None
    jurisdiction: str = Field(min_length=2, max_length=80)
    applicability: str = Field(min_length=8, max_length=500)
    paraphrase: str = Field(min_length=12, max_length=1500)
    allowed_actions: tuple[str, ...] = Field(min_length=1, max_length=16)
    tags: tuple[str, ...] = Field(min_length=1, max_length=24)
    authority: Literal["REGULATOR", "INDUSTRY_GUIDANCE", "MESSAGE_STANDARD"]
    legal_interpretation_prohibited: StrictBool = True

    @field_validator("card_id")
    @classmethod
    def card_id_is_portable(cls, value: str) -> str:
        if CARD_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("card_id must be a lowercase portable identifier")
        return value

    @field_validator("title", "source_title", "jurisdiction", "applicability", "paraphrase")
    @classmethod
    def text_is_trimmed(cls, value: str) -> str:
        if value != value.strip() or not value:
            raise ValueError("policy-card text must be non-empty and trimmed")
        return value

    @field_validator("source_url")
    @classmethod
    def source_url_is_https(cls, value: str) -> str:
        if value != value.strip() or not value.startswith("https://"):
            raise ValueError("source_url must be a trimmed HTTPS URL")
        return value

    @field_validator("effective_from", "effective_to")
    @classmethod
    def effective_times_are_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("effective timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("allowed_actions")
    @classmethod
    def actions_are_known_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("allowed_actions must be unique")
        unknown = sorted(set(value) - VALID_ACTIONS)
        if unknown:
            raise ValueError(f"unknown allowed action(s): {unknown}")
        return value

    @field_validator("tags")
    @classmethod
    def tags_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("tags must be unique")
        for tag in value:
            if CARD_ID_PATTERN.fullmatch(tag) is None:
                raise ValueError("tags must use lowercase portable identifiers")
        return value

    @model_validator(mode="after")
    def effective_interval_is_ordered(self) -> "PolicyCardSource":
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be later than effective_from")
        if not self.legal_interpretation_prohibited:
            raise ValueError("public cards must prohibit autonomous legal interpretation")
        return self


class CompiledPolicyCard(PolicyCardSource):
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    def checksum_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"checksum"})

    def verify_checksum(self) -> None:
        if _card_checksum(self.checksum_payload()) != self.checksum:
            raise ValueError(f"policy card checksum mismatch: {self.card_id}")

    def is_effective_at(self, decision_time: datetime) -> bool:
        if decision_time.tzinfo is None or decision_time.utcoffset() is None:
            raise ValueError("decision_time must be timezone-aware")
        decision_utc = decision_time.astimezone(timezone.utc)
        return self.effective_from <= decision_utc and (
            self.effective_to is None or decision_utc < self.effective_to
        )


class PolicyCardBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["1.0"]
    cards: list[PolicyCardSource] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "PolicyCardBundle":
        ids = [card.card_id for card in self.cards]
        if len(ids) != len(set(ids)):
            raise ValueError("policy card IDs must be unique")
        return self


def compile_policy_cards(source_path: Path, output_dir: Path) -> dict[str, Any]:
    source_path = source_path.resolve()
    raw = require_json_object(strict_json_load(source_path), source=f"policy cards {source_path}")
    bundle = PolicyCardBundle.model_validate(raw)
    compiled: list[CompiledPolicyCard] = []
    for card in sorted(bundle.cards, key=lambda item: item.card_id):
        payload = card.model_dump(mode="json")
        compiled.append(
            CompiledPolicyCard.model_validate({**payload, "checksum": _card_checksum(payload)})
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cards_path = output_dir / "policy_cards.json"
    atomic_json(
        cards_path,
        {
            "schema_version": SCHEMA_VERSION,
            "cards": [card.model_dump(mode="json") for card in compiled],
        },
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "compiled_file": cards_path.name,
        "compiled_sha256": sha256_file(cards_path),
        "card_count": len(compiled),
        "card_ids": [card.card_id for card in compiled],
        "source_urls": sorted({card.source_url for card in compiled}),
        "legal_interpretation_prohibited": True,
    }
    atomic_json(output_dir / "policy_card_manifest.json", manifest)
    return manifest


def load_compiled_policy_cards(path: Path) -> list[CompiledPolicyCard]:
    raw = require_json_object(strict_json_load(path), source=f"compiled policy cards {path}")
    if set(raw) != {"schema_version", "cards"} or raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("compiled policy-card bundle schema is incompatible")
    if not isinstance(raw["cards"], list) or not raw["cards"]:
        raise ValueError("compiled policy-card bundle must contain cards")
    cards = [CompiledPolicyCard.model_validate(item) for item in raw["cards"]]
    ids = [card.card_id for card in cards]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("compiled policy cards must have unique sorted IDs")
    for card in cards:
        card.verify_checksum()
    return cards
