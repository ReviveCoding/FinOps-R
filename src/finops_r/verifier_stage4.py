from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import sha256_file
from .json_safety import require_json_object, strict_json_dumps, strict_json_load
from .planner import ActionPacket
from .retrieval import RetrievalHit
from .verification_policy import VerificationResult

LEARNED_VERIFIER_SCHEMA_VERSION = "1.0"
FEATURE_NAMES = (
    "confidence_shortfall",
    "citation_shortfall",
    "verification_rejected",
    "planner_abstained",
    "retrieval_score_shortfall",
)


class FrozenVerifierArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    verifier_id: str = Field(min_length=1, max_length=160)
    model_type: Literal["LOGISTIC_RISK_SCORER"]
    mode: Literal["VETO_ONLY"]
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    intercept: float
    threshold: float = Field(gt=0.0, lt=1.0)
    training_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def validate_contract(self) -> None:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("learned verifier feature contract is incompatible")
        if len(self.weights) != len(self.feature_names):
            raise ValueError("learned verifier coefficient count is incompatible")


class LearnedVerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    verifier_id: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(gt=0.0, lt=1.0)
    flagged: bool
    mode: Literal["VETO_ONLY"] = "VETO_ONLY"
    feature_values: tuple[float, ...]


def load_frozen_verifier(path: Path) -> tuple[FrozenVerifierArtifact, str]:
    raw = require_json_object(strict_json_load(path), source="frozen verifier artifact")
    artifact = FrozenVerifierArtifact.model_validate(raw)
    artifact.validate_contract()
    return artifact, sha256_file(path)


def verify_frozen_verifier_provenance(
    artifact: FrozenVerifierArtifact,
    *,
    training_path: Path,
    threshold_path: Path,
) -> dict[str, str]:
    training_sha256 = sha256_file(training_path)
    threshold_sha256 = sha256_file(threshold_path)
    if training_sha256 != artifact.training_snapshot_sha256:
        raise ValueError("learned verifier training snapshot hash mismatch")
    if threshold_sha256 != artifact.threshold_snapshot_sha256:
        raise ValueError("learned verifier threshold snapshot hash mismatch")
    return {
        "training_snapshot_sha256": training_sha256,
        "threshold_snapshot_sha256": threshold_sha256,
        "status": "PASS",
    }


def _features(
    packet: ActionPacket,
    verification: VerificationResult,
    hits: list[RetrievalHit],
) -> tuple[float, ...]:
    maximum_score = max((float(hit.score) for hit in hits), default=0.0)
    return (
        1.0 - float(packet.confidence),
        1.0 if not packet.evidence_ids else 0.0,
        1.0 if verification.status == "REJECTED" else 0.0,
        1.0 if packet.abstain else 0.0,
        max(0.0, 1.0 - min(maximum_score, 1.0)),
    )


def run_frozen_verifier(
    packet: ActionPacket,
    verification: VerificationResult,
    hits: list[RetrievalHit],
    *,
    artifact: FrozenVerifierArtifact,
    artifact_sha256: str,
) -> LearnedVerifierResult:
    artifact.validate_contract()
    values = _features(packet, verification, hits)
    logit = artifact.intercept + sum(
        weight * value for weight, value in zip(artifact.weights, values, strict=True)
    )
    if logit >= 0.0:
        exponential = math.exp(-logit)
        probability = 1.0 / (1.0 + exponential)
    else:
        exponential = math.exp(logit)
        probability = exponential / (1.0 + exponential)
    return LearnedVerifierResult(
        verifier_id=artifact.verifier_id,
        artifact_sha256=artifact_sha256,
        risk_probability=probability,
        threshold=artifact.threshold,
        flagged=probability >= artifact.threshold,
        feature_values=values,
    )


def artifact_identity(artifact: FrozenVerifierArtifact) -> str:
    payload: dict[str, Any] = artifact.model_dump(mode="json")
    return hashlib.sha256(
        strict_json_dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
