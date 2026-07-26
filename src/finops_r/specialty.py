from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import STATE_RANK


@dataclass
class RSOPTransformer:
    """Train-only reference-state operational progress features."""

    min_reference_count: int = 30
    epsilon: float = 1e-3
    group_values_: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    state_values_: dict[str, np.ndarray] = field(default_factory=dict)
    global_values_: np.ndarray | None = None
    fitted_: bool = False

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> "RSOPTransformer":
        if len(frame) != len(y):
            raise ValueError("frame and y length mismatch")
        healthy = frame.loc[np.asarray(y) == "NO_EXCEPTION"].copy()
        if len(healthy) < max(self.min_reference_count, 20):
            raise ValueError("insufficient healthy training cases for RSOP")
        self.group_values_ = {}
        for key, group in healthy.groupby(["state", "product"], observed=True):
            values = np.sort(group["time_in_state_min"].dropna().to_numpy(dtype=float))
            if len(values) >= self.min_reference_count:
                self.group_values_[(str(key[0]), str(key[1]))] = values
        self.state_values_ = {}
        for key, group in healthy.groupby("state", observed=True):
            values = np.sort(group["time_in_state_min"].dropna().to_numpy(dtype=float))
            if len(values) >= max(8, self.min_reference_count // 2):
                self.state_values_[str(key)] = values
        self.global_values_ = np.sort(healthy["time_in_state_min"].dropna().to_numpy(dtype=float))
        self.fitted_ = True
        return self

    @staticmethod
    def _percentile(values: np.ndarray, value: float) -> float:
        return float(np.searchsorted(values, value, side="right") / max(1, len(values)))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_ or self.global_values_ is None:
            raise RuntimeError("RSOPTransformer is not fitted")
        result = frame.copy()
        percentiles: list[float] = []
        backoffs: list[float] = []
        for row in frame[["state", "product", "time_in_state_min"]].itertuples(index=False):
            group_key = (str(row.state), str(row.product))
            if group_key in self.group_values_:
                values = self.group_values_[group_key]
                backoff = 0.0
            elif str(row.state) in self.state_values_:
                values = self.state_values_[str(row.state)]
                backoff = 1.0
            else:
                values = self.global_values_
                backoff = 2.0
            value = (
                float(row.time_in_state_min)
                if not pd.isna(row.time_in_state_min)
                else float(np.median(values))
            )
            percentiles.append(self._percentile(values, value))
            backoffs.append(backoff)
        u = np.clip(np.asarray(percentiles), self.epsilon, 1.0 - self.epsilon)
        result["state_rank"] = result["state"].map(STATE_RANK).fillna(-1.0).astype(float)
        result["rsop_percentile"] = u
        result["delay_pressure"] = -np.log(1.0 - u)
        expected_rank = np.clip(result["event_count"].to_numpy(dtype=float) / 10.0, 0.0, 1.0)
        result["path_divergence"] = np.abs(
            result["state_rank"].to_numpy(dtype=float) - expected_rank
        )
        result["reference_backoff_level"] = np.asarray(backoffs)
        return result

    def fit_transform(self, frame: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(frame, y).transform(frame)

    def manifest(self) -> dict[str, object]:
        if not self.fitted_ or self.global_values_ is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "min_reference_count": self.min_reference_count,
            "group_reference_count": len(self.group_values_),
            "state_reference_count": len(self.state_values_),
            "global_reference_rows": len(self.global_values_),
            "adaptation": "PHT-inspired domain extension",
        }
