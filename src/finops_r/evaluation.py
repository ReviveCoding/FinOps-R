from __future__ import annotations

import math
import platform
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    lookup = {label: idx for idx, label in enumerate(classes)}
    encoded = np.asarray([lookup[label] for label in y_true], dtype=int)
    target = np.zeros_like(probabilities)
    target[np.arange(len(encoded)), encoded] = 1.0
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def top_label_ece(
    y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    correct = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(ece)


def wilson_upper(errors: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 1.0
    p = errors / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return float((center + margin) / denominator)


def predictive_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(y_true, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, predictions, labels=classes, average="weighted", zero_division=0)
        ),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "multiclass_brier": multiclass_brier(y_true, probabilities, classes),
        "top_label_ece": top_label_ece(y_true, predictions, probabilities),
    }


def slice_metrics(frame: pd.DataFrame, predictions: np.ndarray, candidate_id: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for slice_name in ("product", "state", "critical", "evidence_complete", "generator_family"):
        values = frame[slice_name].astype(str)
        for value in sorted(values.unique()):
            mask = values == value
            if int(mask.sum()) < 8:
                continue
            records.append(
                {
                    "candidate_id": candidate_id,
                    "slice_name": slice_name,
                    "slice_value": value,
                    "support": int(mask.sum()),
                    "macro_f1": float(
                        f1_score(
                            frame.loc[mask, "root_cause"],
                            predictions[mask],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "error_rate": float(
                        np.mean(predictions[mask] != frame.loc[mask, "root_cause"].to_numpy())
                    ),
                }
            )
    return pd.DataFrame(records)


def benchmark_latency(
    predict_fn: Callable[[pd.DataFrame], Any], frame: pd.DataFrame, repeats: int = 5
) -> dict[str, float]:
    if frame.empty:
        raise ValueError("latency benchmark requires at least one case")
    if repeats < 1:
        raise ValueError("latency benchmark requires at least one repeat")
    sample = frame.iloc[: min(128, len(frame))]
    timings = []
    predict_fn(sample)
    for _ in range(repeats):
        started = time.perf_counter()
        predict_fn(sample)
        timings.append((time.perf_counter() - started) * 1000.0 / len(sample))
    array = np.asarray(timings)
    return {
        "latency_ms_per_case_p50": float(np.percentile(array, 50)),
        "latency_ms_per_case_p95": float(np.percentile(array, 95)),
        "latency_ms_per_case_p99": float(np.percentile(array, 99)),
        "throughput_cases_per_second": float(1000.0 / max(1e-9, np.mean(array))),
    }


def paired_bootstrap_delta(
    baseline_cost: np.ndarray,
    candidate_cost: np.ndarray,
    seed: int,
    resamples: int,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    baseline_cost = np.asarray(baseline_cost, dtype=float)
    candidate_cost = np.asarray(candidate_cost, dtype=float)
    if len(baseline_cost) != len(candidate_cost):
        raise ValueError("paired costs must have equal length")
    if baseline_cost.ndim != 1 or candidate_cost.ndim != 1:
        raise ValueError("paired costs must be one-dimensional")
    if not np.isfinite(baseline_cost).all() or not np.isfinite(candidate_cost).all():
        raise ValueError("paired costs must be finite")
    if len(baseline_cost) == 0 or isinstance(resamples, bool) or resamples < 1:
        raise ValueError("paired bootstrap requires data and at least one resample")
    rng = np.random.default_rng(seed)
    clusters, resampling_unit = _bootstrap_clusters(len(baseline_cost), groups)
    delta = baseline_cost - candidate_cost
    draws = np.empty(resamples, dtype=float)
    for idx in range(resamples):
        sample = _draw_cluster_sample(rng, clusters)
        draws[idx] = float(delta[sample].mean())
    return {
        "mean_harm_reduction_per_case": float(delta.mean()),
        "ci95_lower": float(np.percentile(draws, 2.5)),
        "ci95_upper": float(np.percentile(draws, 97.5)),
        "probability_improvement": float(np.mean(draws > 0)),
        "resamples": int(resamples),
        "resampling_unit": resampling_unit,
        "cluster_count": len(clusters),
    }


def paired_bootstrap_macro_f1_delta(
    y_true: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    seed: int,
    resamples: int,
    classes: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    baseline_predictions = np.asarray(baseline_predictions)
    candidate_predictions = np.asarray(candidate_predictions)
    if not (len(y_true) == len(baseline_predictions) == len(candidate_predictions)):
        raise ValueError("paired classification arrays must have equal length")
    if any(array.ndim != 1 for array in (y_true, baseline_predictions, candidate_predictions)):
        raise ValueError("paired classification arrays must be one-dimensional")
    if len(y_true) == 0 or isinstance(resamples, bool) or resamples < 1:
        raise ValueError("paired bootstrap requires data and at least one resample")
    if classes is None:
        fixed_labels = np.unique(
            np.concatenate((y_true, baseline_predictions, candidate_predictions))
        )
    else:
        fixed_labels = np.asarray(classes)
        if fixed_labels.ndim != 1 or len(fixed_labels) == 0:
            raise ValueError("classes must be a non-empty one-dimensional array")
        if len(np.unique(fixed_labels)) != len(fixed_labels):
            raise ValueError("classes must not contain duplicates")
        observed_labels = set(np.concatenate((y_true, baseline_predictions, candidate_predictions)))
        if observed_labels - set(fixed_labels):
            raise ValueError("classification arrays contain labels outside classes")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    clusters, resampling_unit = _bootstrap_clusters(len(y_true), groups)
    for idx in range(resamples):
        sample = _draw_cluster_sample(rng, clusters)
        baseline = f1_score(
            y_true[sample],
            baseline_predictions[sample],
            labels=fixed_labels,
            average="macro",
            zero_division=0,
        )
        candidate = f1_score(
            y_true[sample],
            candidate_predictions[sample],
            labels=fixed_labels,
            average="macro",
            zero_division=0,
        )
        draws[idx] = float(candidate - baseline)
    observed = float(
        f1_score(
            y_true, candidate_predictions, labels=fixed_labels, average="macro", zero_division=0
        )
        - f1_score(
            y_true, baseline_predictions, labels=fixed_labels, average="macro", zero_division=0
        )
    )
    return {
        "observed_macro_f1_delta": observed,
        "ci95_lower": float(np.percentile(draws, 2.5)),
        "ci95_upper": float(np.percentile(draws, 97.5)),
        "probability_improvement": float(np.mean(draws > 0)),
        "resamples": int(resamples),
        "resampling_unit": resampling_unit,
        "cluster_count": len(clusters),
        "label_count": len(fixed_labels),
    }


def _bootstrap_clusters(
    n_rows: int, groups: np.ndarray | None
) -> tuple[tuple[np.ndarray, ...], str]:
    if groups is None:
        return tuple(np.asarray([index], dtype=int) for index in range(n_rows)), "case"
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or len(group_values) != n_rows:
        raise ValueError("bootstrap groups must contain one value per row")
    if pd.isna(group_values).any():
        raise ValueError("bootstrap groups must not contain missing values")
    codes, unique_groups = pd.factorize(group_values, sort=True)
    clusters = tuple(np.flatnonzero(codes == index) for index in range(len(unique_groups)))
    if not clusters:
        raise ValueError("bootstrap groups must not be empty")
    return clusters, "group"


def _draw_cluster_sample(rng: np.random.Generator, clusters: tuple[np.ndarray, ...]) -> np.ndarray:
    selected = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[int(index)] for index in selected])


def confusion_records(
    y_true: np.ndarray, predictions: np.ndarray, classes: np.ndarray
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, predictions, labels=classes)
    records = []
    for row_idx, true_label in enumerate(classes):
        for col_idx, predicted_label in enumerate(classes):
            records.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[row_idx, col_idx]),
                }
            )
    return pd.DataFrame(records)


def hardware_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
