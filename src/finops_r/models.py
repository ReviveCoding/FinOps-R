from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from .constants import CAUSES
from .preprocessing import build_preprocessor, select_features


class RuleSanityClassifier:
    # scikit-learn metric contracts expect class-probability columns in lexical label order.
    classes_ = np.asarray(sorted(CAUSES))

    @staticmethod
    def _predict_one(row: pd.Series) -> str:
        checks = (
            ("COMPLIANCE_HOLD", (row.get("compliance_score", 0) or 0) >= 0.68),
            ("FUNDING_SHORTAGE", (row.get("funding_ratio", 1.0) or 1.0) < 0.92),
            ("SECURITIES_SHORTAGE", (row.get("securities_ratio", 1.0) or 1.0) < 0.92),
            ("SECURITY_ID_MISMATCH", row.get("security_id_match", 1.0) == 0),
            ("CURRENCY_MISMATCH", row.get("currency_match", 1.0) == 0),
            ("COUNTERPARTY_MISMATCH", row.get("counterparty_match", 1.0) == 0),
            ("MISSING_SSI", row.get("ssi_present", 1.0) == 0),
            ("STALE_SSI", (row.get("ssi_version_gap", 0) or 0) >= 1),
            ("DUPLICATE_AMENDMENT", (row.get("duplicate_amendment_count", 0) or 0) >= 2),
            ("QUANTITY_MISMATCH", (row.get("qty_diff_pct", 0) or 0) >= 0.025),
            ("PRICE_MISMATCH", (row.get("price_diff_bps", 0) or 0) >= 25),
            (
                "LATE_ALLOCATION",
                row.get("allocation_present", 1.0) == 0
                and row.get("state") in {"EXECUTED", "ALLOCATED"},
            ),
            (
                "MISSING_AFFIRMATION",
                row.get("affirmation_present", 1.0) == 0 and row.get("state") == "CONFIRMED",
            ),
        )
        for cause, condition in checks:
            if condition:
                return cause
        return "NO_EXCEPTION"

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> "RuleSanityClassifier":
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([self._predict_one(row) for _, row in frame.iterrows()])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        predicted = self.predict(frame)
        probabilities = np.full(
            (len(frame), len(self.classes_)), 0.01 / (len(self.classes_) - 1), dtype=float
        )
        lookup = {label: idx for idx, label in enumerate(self.classes_)}
        for row_idx, label in enumerate(predicted):
            probabilities[row_idx, lookup[label]] = 0.99
        return probabilities


@dataclass
class TabularCandidate:
    candidate_id: str
    use_rsop: bool
    advanced_ensemble: bool
    config: dict[str, Any]
    preprocessor: Any = None
    estimators: list[Any] = field(default_factory=list)
    weights: np.ndarray | None = None
    classes_: np.ndarray | None = None
    fit_seconds_: float = 0.0

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> "TabularCandidate":
        started = time.perf_counter()
        self.preprocessor = build_preprocessor(self.use_rsop)
        matrix = self.preprocessor.fit_transform(select_features(frame, self.use_rsop))
        ensemble_cfg = self.config["ensemble"]
        hist = HistGradientBoostingClassifier(
            learning_rate=float(ensemble_cfg["hist_learning_rate"]),
            max_iter=int(ensemble_cfg["hist_max_iter"]),
            max_leaf_nodes=int(ensemble_cfg["hist_max_leaf_nodes"]),
            l2_regularization=0.2,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=int(self.config["seed"]),
        )
        hist.fit(matrix, y)
        self.estimators = [hist]
        self.weights = np.asarray([1.0])
        if self.advanced_ensemble:
            extra = ExtraTreesClassifier(
                n_estimators=int(ensemble_cfg["extra_trees_estimators"]),
                min_samples_leaf=int(ensemble_cfg["extra_trees_min_samples_leaf"]),
                max_features="sqrt",
                class_weight="balanced",
                # One process keeps Windows, CI, and sandbox behavior reproducible.
                # Users can raise this through a future profiled config rather than implicit fan-out.
                n_jobs=1,
                random_state=int(self.config["seed"]) + 101,
            )
            extra.fit(matrix, y)
            self.estimators.append(extra)
            self.weights = np.asarray(
                [float(ensemble_cfg["weight_hist"]), float(ensemble_cfg["weight_extra"])],
                dtype=float,
            )
            self.weights /= self.weights.sum()
        self.classes_ = np.asarray(hist.classes_)
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("candidate is not fitted")
        return self.preprocessor.transform(select_features(frame, self.use_rsop))

    def component_probabilities(self, frame: pd.DataFrame) -> list[np.ndarray]:
        matrix = self._matrix(frame)
        return [estimator.predict_proba(matrix) for estimator in self.estimators]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("candidate is not fitted")
        components = self.component_probabilities(frame)
        result = np.zeros_like(components[0], dtype=float)
        for weight, probabilities in zip(self.weights, components, strict=True):
            result += weight * probabilities
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("candidate is not fitted")
        probabilities = self.predict_proba(frame)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def disagreement(self, frame: pd.DataFrame) -> np.ndarray:
        components = self.component_probabilities(frame)
        if len(components) == 1:
            return np.zeros(len(frame), dtype=float)
        stacked = np.stack(components, axis=0)
        mean = stacked.mean(axis=0)
        return np.mean(np.abs(stacked - mean[None, :, :]), axis=(0, 2))


@dataclass
class TemperatureScaler:
    lower: float = 0.35
    upper: float = 4.0
    temperature_: float = 1.0
    classes_: np.ndarray | None = None

    @staticmethod
    def _apply(probabilities: np.ndarray, temperature: float) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def fit(
        self, probabilities: np.ndarray, y: np.ndarray, classes: np.ndarray
    ) -> "TemperatureScaler":
        self.classes_ = np.asarray(classes)
        index = {label: idx for idx, label in enumerate(self.classes_)}
        target = np.asarray([index[label] for label in y], dtype=int)

        def objective(value: float) -> float:
            calibrated = self._apply(probabilities, float(value))
            return float(
                -np.mean(np.log(np.clip(calibrated[np.arange(len(target)), target], 1e-12, 1.0)))
            )

        result = minimize_scalar(objective, bounds=(self.lower, self.upper), method="bounded")
        self.temperature_ = float(result.x if result.success else 1.0)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return self._apply(probabilities, self.temperature_)


def build_candidate(
    candidate_id: str, config: dict[str, Any]
) -> RuleSanityClassifier | TabularCandidate:
    if candidate_id == "C0":
        return RuleSanityClassifier()
    definitions = {
        "C1": (False, False),
        "C2": (False, True),
        "C3": (True, False),
        "C4": (True, True),
    }
    if candidate_id not in definitions:
        raise ValueError(f"unknown trainable candidate: {candidate_id}")
    use_rsop, advanced = definitions[candidate_id]
    return TabularCandidate(candidate_id, use_rsop, advanced, config)
