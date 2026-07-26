from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from finops_r.artifacts import atomic_json, sha256_file
from finops_r.json_safety import require_json_object, strict_json_load
from finops_r.verifier_stage4 import FEATURE_NAMES


def _load_records(path: Path, expected_purpose: str) -> list[tuple[list[float], int]]:
    raw = require_json_object(strict_json_load(path), source=f"verifier snapshot {path}")
    if set(raw) != {"schema_version", "purpose", "feature_names", "records"}:
        raise ValueError("verifier snapshot schema is incompatible")
    if raw["schema_version"] != "1.0" or raw["purpose"] != expected_purpose:
        raise ValueError("verifier snapshot identity is incompatible")
    if tuple(raw["feature_names"]) != FEATURE_NAMES:
        raise ValueError("verifier snapshot feature contract is incompatible")
    records = raw["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("verifier snapshot records must be non-empty")
    identifiers: set[str] = set()
    parsed: list[tuple[list[float], int]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "record_id",
            "features",
            "unsafe",
        }:
            raise ValueError("verifier training record schema is incompatible")
        identifier = record["record_id"]
        features = record["features"]
        label = record["unsafe"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or not isinstance(features, list)
            or len(features) != len(FEATURE_NAMES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in features
            )
            or isinstance(label, bool)
            or label not in (0, 1)
        ):
            raise ValueError("verifier training record contract is invalid")
        identifiers.add(identifier)
        parsed.append(([float(value) for value in features], int(label)))
    if {label for _, label in parsed} != {0, 1}:
        raise ValueError("verifier snapshot must contain both labels")
    return parsed


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _fit(records: list[tuple[list[float], int]]) -> tuple[list[float], float]:
    weights = [0.0] * len(FEATURE_NAMES)
    intercept = 0.0
    learning_rate = 0.15
    regularization = 0.02
    for _ in range(4000):
        weight_gradient = [0.0] * len(weights)
        intercept_gradient = 0.0
        for features, label in records:
            score = _sigmoid(
                intercept
                + sum(weight * feature for weight, feature in zip(weights, features, strict=True))
            )
            error = score - label
            intercept_gradient += error
            for index, feature in enumerate(features):
                weight_gradient[index] += error * feature
        count = float(len(records))
        intercept -= learning_rate * intercept_gradient / count
        for index in range(len(weights)):
            gradient = weight_gradient[index] / count + regularization * weights[index]
            weights[index] -= learning_rate * gradient
    return [round(value, 12) for value in weights], round(intercept, 12)


def _score(features: list[float], weights: list[float], intercept: float) -> float:
    return _sigmoid(
        intercept + sum(weight * feature for weight, feature in zip(weights, features, strict=True))
    )


def _select_threshold(
    records: list[tuple[list[float], int]],
    weights: list[float],
    intercept: float,
) -> float:
    scored = [(_score(features, weights, intercept), label) for features, label in records]
    candidates = sorted({0.01, 0.99, *(score for score, _ in scored)})
    eligible: list[tuple[int, float]] = []
    for threshold in candidates:
        false_negatives = sum(label == 1 and score < threshold for score, label in scored)
        false_positives = sum(label == 0 and score >= threshold for score, label in scored)
        if false_negatives == 0:
            eligible.append((false_positives, threshold))
    if not eligible:
        raise RuntimeError("no threshold satisfies the zero-false-negative constraint")
    _, selected = min(eligible, key=lambda item: (item[0], -item[1]))
    return round(selected, 12)


def freeze(
    *,
    training_path: Path,
    threshold_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    training = _load_records(training_path, "DEVELOPMENT_TRAINING_ONLY")
    threshold_records = _load_records(
        threshold_path,
        "DEVELOPMENT_THRESHOLD_SELECTION_ONLY",
    )
    weights, intercept = _fit(training)
    threshold = _select_threshold(threshold_records, weights, intercept)
    artifact = {
        "schema_version": "1.0",
        "verifier_id": "finops_learned_risk_verifier_stage4_v1",
        "model_type": "LOGISTIC_RISK_SCORER",
        "mode": "VETO_ONLY",
        "feature_names": list(FEATURE_NAMES),
        "weights": weights,
        "intercept": intercept,
        "threshold": threshold,
        "training_snapshot_sha256": sha256_file(training_path),
        "threshold_snapshot_sha256": sha256_file(threshold_path),
    }
    atomic_json(output_path, artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the deterministic FinOps-R Stage 4 learned verifier"
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=Path("data/policy_cards/learned_verifier_training_stage4.json"),
    )
    parser.add_argument(
        "--threshold",
        type=Path,
        default=Path("data/policy_cards/learned_verifier_threshold_stage4.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/policy_cards/frozen_learned_verifier_stage4.json"),
    )
    args = parser.parse_args()
    freeze(
        training_path=args.training,
        threshold_path=args.threshold,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
