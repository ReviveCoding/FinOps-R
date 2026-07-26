from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .constants import CRITICAL_CAUSES, HARM_BY_CAUSE


ReviewPolicy = Literal["fifo", "confidence", "disagreement", "evr"]


def route_decisions(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    policy: ReviewPolicy,
    review_capacity: float,
    human_correction_probability: float,
    review_cost: float,
    disagreement: np.ndarray | None = None,
    envelopes: pd.DataFrame | None = None,
    minimum_review_score: float | None = None,
) -> pd.DataFrame:
    n_rows = len(frame)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities, dtype=float)
    if n_rows == 0:
        raise ValueError("decision routing requires at least one case")
    if predictions.ndim != 1 or len(predictions) != n_rows:
        raise ValueError("predictions must contain exactly one label per case")
    if probabilities.ndim != 2 or probabilities.shape[0] != n_rows or probabilities.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional case-by-class matrix")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("probabilities must be finite, bounded, and row-normalized")
    if not np.isfinite(review_capacity) or not 0.0 <= review_capacity <= 1.0:
        raise ValueError("review_capacity must be between zero and one")
    if (
        not np.isfinite(human_correction_probability)
        or not 0.0 <= human_correction_probability <= 1.0
    ):
        raise ValueError("human_correction_probability must be between zero and one")
    if not np.isfinite(review_cost) or review_cost < 0.0:
        raise ValueError("review_cost must be finite and non-negative")
    if minimum_review_score is not None and not np.isfinite(minimum_review_score):
        raise ValueError("minimum_review_score must be finite")
    if disagreement is not None:
        disagreement = np.asarray(disagreement, dtype=float)
        if (
            disagreement.ndim != 1
            or len(disagreement) != n_rows
            or not np.isfinite(disagreement).all()
            or (disagreement < 0.0).any()
        ):
            raise ValueError("disagreement must be one finite non-negative value per case")
    if envelopes is not None and (
        len(envelopes) != n_rows
        or not {"policy_status", "verifier_status"}.issubset(envelopes.columns)
    ):
        raise ValueError("envelopes must contain one complete gate record per case")
    confidence = probabilities.max(axis=1)
    error_risk = 1.0 - confidence
    try:
        predicted_harm = np.asarray(
            [HARM_BY_CAUSE[str(label)] for label in predictions], dtype=float
        )
    except KeyError as exc:
        raise ValueError(f"unknown predicted cause: {exc.args[0]}") from exc
    hard_hold = np.zeros(n_rows, dtype=bool)
    reason = np.full(n_rows, "LOW_RISK_AUTO", dtype=object)
    if envelopes is not None:
        hard_hold = (envelopes["policy_status"].to_numpy() != "PASS") | (
            envelopes["verifier_status"].to_numpy() != "SUPPORTED"
        )
        reason[hard_hold] = "HARD_GATE_HOLD"
    if policy == "fifo":
        score = (n_rows - np.arange(n_rows, dtype=float)) / max(1, n_rows)
    elif policy == "confidence":
        score = error_risk
    elif policy == "disagreement":
        score = error_risk + (np.zeros(n_rows) if disagreement is None else disagreement)
    elif policy == "evr":
        critical_observables = [
            "ssi_version_gap",
            "qty_diff_pct",
            "price_diff_bps",
            "funding_ratio",
            "securities_ratio",
            "compliance_score",
        ]
        missing = sorted(set(critical_observables + ["evidence_complete"]) - set(frame.columns))
        if missing:
            raise ValueError(f"EVR inputs are missing required observables: {missing}")
        data_quality_risk = frame[critical_observables].isna().mean(axis=1).to_numpy(dtype=float)
        data_quality_risk += 0.5 * (
            1.0 - frame["evidence_complete"].fillna(0.0).to_numpy(dtype=float)
        )
        combined_error_risk = np.clip(error_risk + 0.85 * data_quality_risk, 0.0, 1.0)
        harm_scale = max(HARM_BY_CAUSE.values())
        normalized_harm = predicted_harm / max(harm_scale, 1e-12)
        expected_avoidable_harm = (
            combined_error_risk * (0.50 + 0.50 * normalized_harm) * human_correction_probability
        )
        score = expected_avoidable_harm + 0.25 * data_quality_risk - review_cost
        if disagreement is not None:
            score += disagreement
    else:
        raise ValueError(f"unsupported review policy: {policy}")
    eligible = np.where(~hard_hold)[0]
    manual_budget = int(np.floor(review_capacity * n_rows))
    review_budget = min(len(eligible), max(0, manual_budget - int(hard_hold.sum())))
    if policy == "evr":
        eligible = eligible[score[eligible] > 0.0]
        review_budget = min(review_budget, len(eligible))
    if minimum_review_score is not None:
        eligible = eligible[score[eligible] > minimum_review_score]
        review_budget = min(review_budget, len(eligible))
    selected = (
        eligible[np.argsort(score[eligible])[::-1][:review_budget]]
        if review_budget
        else np.asarray([], dtype=int)
    )
    route = np.full(n_rows, "SIMULATE", dtype=object)
    route[hard_hold] = "HOLD"
    route[selected] = "REVIEW"
    reason[selected] = f"{policy.upper()}_REVIEW"
    return pd.DataFrame(
        {
            "case_id": frame["case_id"].to_numpy(),
            "route": route,
            "route_reason": reason,
            "confidence": confidence,
            "calibrated_error_risk": error_risk,
            "review_score": score,
        },
        index=frame.index,
    )


def fit_review_threshold(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    policy: ReviewPolicy,
    review_capacity: float,
    human_correction_probability: float,
    review_cost: float,
    disagreement: np.ndarray | None = None,
    envelopes: pd.DataFrame | None = None,
) -> float:
    """Fit a deployment-stable review threshold on the dedicated policy split."""
    routed = route_decisions(
        frame,
        predictions,
        probabilities,
        policy,
        review_capacity,
        human_correction_probability,
        review_cost,
        disagreement,
        envelopes,
    )
    reviewed = routed.loc[routed["route"] == "REVIEW", "review_score"]
    if not reviewed.empty:
        boundary = float(reviewed.min())
        simulated = routed.loc[routed["route"] == "SIMULATE", "review_score"].to_numpy(dtype=float)
        # A scalar online cutoff cannot select only some cases tied at the
        # boundary. Exclude the full tie rather than exceed human capacity.
        if np.any(simulated == boundary):
            return boundary
        return float(np.nextafter(boundary, -np.inf))
    scores = routed["review_score"].to_numpy(dtype=float)
    return float(scores.max()) if len(scores) else 1.0


def decision_metrics(
    frame: pd.DataFrame,
    proposed_actions: np.ndarray,
    routes: pd.DataFrame,
    human_correction_probability: float,
    review_cost: float,
    hold_cost_fraction: float = 0.10,
    hold_fixed_cost: float = 0.10,
) -> tuple[dict[str, float], np.ndarray]:
    incorrect = proposed_actions != frame["oracle_action"].to_numpy()
    harm = frame["harm"].to_numpy(dtype=float)
    route_values = routes["route"].to_numpy()
    residual = np.zeros(len(frame), dtype=float)
    simulate = route_values == "SIMULATE"
    review = route_values == "REVIEW"
    hold = route_values == "HOLD"
    residual[simulate & incorrect] = harm[simulate & incorrect]
    residual[review & incorrect] = harm[review & incorrect] * (1.0 - human_correction_probability)
    residual[review] += review_cost
    residual[hold] = hold_cost_fraction * harm[hold] + hold_fixed_cost
    critical = frame["root_cause"].isin(CRITICAL_CAUSES).to_numpy()
    false_greenlight = critical & incorrect & simulate
    critical_rate = float(false_greenlight.sum() / max(1, critical.sum()))
    metrics = {
        "residual_harm_per_1000": float(residual.mean() * 1000.0),
        "unsafe_automated_harm_per_1000": float(
            np.sum(harm[simulate & incorrect]) / len(frame) * 1000.0
        ),
        "wrong_automated_action_rate": float(np.mean(simulate & incorrect)),
        "review_rate": float(review.mean()),
        "hold_rate": float(hold.mean()),
        "manual_intervention_rate": float(np.mean(review | hold)),
        "automation_rate": float(simulate.mean()),
        "critical_false_greenlight_rate": critical_rate,
        "critical_case_count": int(critical.sum()),
        "critical_false_greenlight_count": int(false_greenlight.sum()),
        "mean_case_cost": float(residual.mean()),
    }
    return metrics, residual
