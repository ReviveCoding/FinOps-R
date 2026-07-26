from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    ACTION_BY_CAUSE,
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CAUSES,
    CRITICAL_CAUSES,
    HARM_BY_CAUSE,
    STATES,
)


CAUSE_PROBABILITIES = np.array(
    [0.25, 0.06, 0.05, 0.05, 0.06, 0.06, 0.04, 0.08, 0.07, 0.05, 0.04, 0.05, 0.05, 0.04],
    dtype=float,
)
CAUSE_PROBABILITIES /= CAUSE_PROBABILITIES.sum()

STATE_TIME_SCALE = {
    "EXECUTED": 28.0,
    "ALLOCATED": 38.0,
    "CONFIRMED": 45.0,
    "INSTRUCTED": 52.0,
    "MATCHED": 32.0,
}
PRODUCT_TIME_FACTOR = {"EQUITY": 0.85, "ETF": 0.90, "CORP_BOND": 1.35, "OPTION": 1.10}
VALID_SPLITS = {"train", "validation", "calibration", "policy", "test", "stress"}
VALID_CATEGORIES = {
    "state": set(STATES),
    "product": {"EQUITY", "ETF", "CORP_BOND", "OPTION"},
    "route": {"DTC", "FEDWIRE", "EUROCLEAR", "UNKNOWN"},
    "currency": {"USD", "EUR", "GBP"},
    "side": {"BUY", "SELL"},
    "generator_family": {"CORE_A", "CORE_B", "STRESS_C"},
}
NULLABLE_OBSERVABLES = {
    "ssi_version_gap",
    "qty_diff_pct",
    "price_diff_bps",
    "funding_ratio",
    "securities_ratio",
    "compliance_score",
}
BINARY_COLUMNS = {
    "allocation_present",
    "affirmation_present",
    "ssi_present",
    "currency_match",
    "counterparty_match",
    "security_id_match",
    "evidence_complete",
    "critical",
}
NON_NEGATIVE_COLUMNS = {
    "time_in_state_min",
    "case_age_min",
    "event_count",
    "artifact_count",
    "missing_doc_count",
    "ssi_version_gap",
    "qty_diff_pct",
    "price_diff_bps",
    "duplicate_amendment_count",
    "funding_ratio",
    "securities_ratio",
    "compliance_score",
    "harm",
}
INTEGER_COLUMNS = {"event_count", "artifact_count", "missing_doc_count"}
REQUIRED_CASE_COLUMNS = {
    "case_id",
    "trade_id",
    "entity_id",
    "decision_time",
    "split",
    "generator_family",
    "root_cause",
    "oracle_action",
    "critical",
    "harm",
    *CATEGORICAL_FEATURES,
    *BASE_NUMERIC_FEATURES,
}


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def entity_split(entity_id: str) -> str:
    bucket = int(stable_hash(entity_id)[:8], 16) % 1000
    if bucket < 550:
        return "train"
    if bucket < 650:
        return "validation"
    if bucket < 750:
        return "calibration"
    if bucket < 850:
        return "policy"
    if bucket < 950:
        return "test"
    return "stress"


def _base_row(
    rng: np.random.Generator, case_idx: int, entity_id: str, split: str
) -> dict[str, Any]:
    state = str(rng.choice(STATES, p=[0.14, 0.18, 0.23, 0.26, 0.19]))
    product = str(rng.choice(["EQUITY", "ETF", "CORP_BOND", "OPTION"], p=[0.42, 0.16, 0.27, 0.15]))
    route = str(rng.choice(["DTC", "FEDWIRE", "EUROCLEAR"], p=[0.63, 0.15, 0.22]))
    currency = str(rng.choice(["USD", "EUR", "GBP"], p=[0.78, 0.14, 0.08]))
    side = str(rng.choice(["BUY", "SELL"]))
    time_scale = STATE_TIME_SCALE[state] * PRODUCT_TIME_FACTOR[product]
    time_in_state = float(rng.gamma(shape=2.1, scale=time_scale / 2.1))
    state_idx = STATES.index(state)
    case_age = float(time_in_state + state_idx * rng.uniform(28.0, 70.0) + rng.uniform(0, 25))
    deadline = float(300.0 - case_age + rng.normal(0, 25))
    event_count = int(max(1, state_idx * 2 + rng.poisson(2)))
    artifact_count = max(1, event_count + int(rng.integers(0, 4)))
    return {
        "case_id": f"T{case_idx:07d}",
        "trade_id": f"TR{case_idx:07d}",
        "entity_id": entity_id,
        "decision_time": (
            datetime(2027, 1, 5, tzinfo=timezone.utc) + timedelta(minutes=case_idx)
        ).isoformat(),
        "split": split,
        "generator_family": "STRESS_C"
        if split == "stress"
        else str(rng.choice(["CORE_A", "CORE_B"])),
        "state": state,
        "product": product,
        "route": route,
        "currency": currency,
        "side": side,
        "time_in_state_min": time_in_state,
        "case_age_min": case_age,
        "deadline_slack_min": deadline,
        "event_count": event_count,
        "artifact_count": artifact_count,
        "missing_doc_count": int(rng.binomial(2, 0.04)),
        "ssi_version_gap": float(rng.binomial(1, 0.025)),
        "qty_diff_pct": float(abs(rng.normal(0, 0.003))),
        "price_diff_bps": float(abs(rng.normal(0, 3.0))),
        "duplicate_amendment_count": float(rng.binomial(1, 0.025)),
        "funding_ratio": float(np.clip(rng.normal(1.06, 0.07), 0.55, 1.35)),
        "securities_ratio": float(np.clip(rng.normal(1.06, 0.07), 0.55, 1.35)),
        "compliance_score": float(np.clip(rng.beta(1.3, 9.0), 0, 1)),
        "allocation_present": 1.0,
        "affirmation_present": 1.0,
        "ssi_present": 1.0,
        "currency_match": 1.0,
        "counterparty_match": 1.0,
        "security_id_match": 1.0,
        "evidence_complete": float(rng.random() > 0.06),
    }


def _inject_cause(row: dict[str, Any], cause: str, rng: np.random.Generator, stress: bool) -> None:
    severity_noise = 1.35 if stress else 1.0
    if cause == "STALE_SSI":
        row["state"] = str(rng.choice(["INSTRUCTED", "MATCHED"]))
        row["ssi_version_gap"] = float(rng.integers(1, 5))
    elif cause == "MISSING_SSI":
        row["state"] = "INSTRUCTED"
        row["ssi_present"] = 0.0
        row["missing_doc_count"] += int(rng.integers(1, 4))
    elif cause == "COUNTERPARTY_MISMATCH":
        row["counterparty_match"] = 0.0
    elif cause == "QUANTITY_MISMATCH":
        row["qty_diff_pct"] = float(abs(rng.normal(0.065, 0.025)) * severity_noise)
    elif cause == "PRICE_MISMATCH":
        row["price_diff_bps"] = float(abs(rng.normal(68, 25)) * severity_noise)
    elif cause == "CURRENCY_MISMATCH":
        row["currency_match"] = 0.0
    elif cause == "LATE_ALLOCATION":
        row["state"] = str(rng.choice(["EXECUTED", "ALLOCATED"]))
        row["allocation_present"] = 0.0
        row["time_in_state_min"] *= float(rng.uniform(2.4, 4.2))
        row["deadline_slack_min"] -= float(rng.uniform(70, 170))
    elif cause == "MISSING_AFFIRMATION":
        row["state"] = "CONFIRMED"
        row["affirmation_present"] = 0.0
        row["time_in_state_min"] *= float(rng.uniform(2.0, 3.7))
        row["deadline_slack_min"] -= float(rng.uniform(55, 150))
    elif cause == "SECURITY_ID_MISMATCH":
        row["security_id_match"] = 0.0
    elif cause == "DUPLICATE_AMENDMENT":
        row["duplicate_amendment_count"] = float(rng.integers(2, 5))
    elif cause == "FUNDING_SHORTAGE":
        row["funding_ratio"] = float(rng.uniform(0.55, 0.91))
    elif cause == "SECURITIES_SHORTAGE":
        row["securities_ratio"] = float(rng.uniform(0.55, 0.91))
    elif cause == "COMPLIANCE_HOLD":
        row["compliance_score"] = float(rng.uniform(0.72, 1.0))
    row["case_age_min"] = max(float(row["case_age_min"]), float(row["time_in_state_min"]))


def _inject_observation_noise(row: dict[str, Any], rng: np.random.Generator, stress: bool) -> None:
    flip_p = 0.035 if not stress else 0.11
    if rng.random() < flip_p:
        flag = str(
            rng.choice(
                [
                    "allocation_present",
                    "affirmation_present",
                    "ssi_present",
                    "currency_match",
                    "counterparty_match",
                    "security_id_match",
                ]
            )
        )
        row[flag] = 1.0 - float(row[flag])
    missing_p = 0.018 if not stress else 0.075
    for key in (
        "ssi_version_gap",
        "qty_diff_pct",
        "price_diff_bps",
        "funding_ratio",
        "securities_ratio",
        "compliance_score",
    ):
        if rng.random() < missing_p:
            row[key] = np.nan
    if stress and rng.random() < 0.25:
        row["evidence_complete"] = 0.0
        row["missing_doc_count"] += int(rng.integers(1, 3))
    if rng.random() < (0.01 if not stress else 0.05):
        row["route"] = "UNKNOWN"


def generate_cases(n_cases: int, n_entities: int, seed: int) -> pd.DataFrame:
    for name, value in (("n_cases", n_cases), ("n_entities", n_entities), ("seed", seed)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if n_cases < 200:
        raise ValueError("n_cases must be at least 200 for the candidate study")
    if n_entities < 100:
        raise ValueError("n_entities must be at least 100")
    if n_entities > n_cases:
        raise ValueError("n_entities cannot exceed n_cases")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for idx in range(n_cases):
        entity_id = f"ENT{idx % n_entities:06d}"
        split = entity_split(entity_id)
        stress = split == "stress"
        probabilities = CAUSE_PROBABILITIES.copy()
        if stress:
            probabilities[0] *= 0.55
            probabilities[11:] *= 1.65
            probabilities /= probabilities.sum()
        cause = str(rng.choice(CAUSES, p=probabilities))
        row = _base_row(rng, idx, entity_id, split)
        _inject_cause(row, cause, rng, stress)
        _inject_observation_noise(row, rng, stress)
        row["root_cause"] = cause
        row["oracle_action"] = ACTION_BY_CAUSE[cause]
        row["critical"] = int(cause in CRITICAL_CAUSES)
        row["harm"] = HARM_BY_CAUSE[cause]
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values("case_id").reset_index(drop=True)


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    normalized = frame.sort_values("case_id").to_json(
        orient="records", date_format="iso", double_precision=12
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _is_timezone_aware(value: object) -> bool:
    try:
        return pd.Timestamp(value).tzinfo is not None
    except (TypeError, ValueError):
        return False


def validate_cases(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate the complete training contract and report every detected violation."""
    duplicate_column_count = int(frame.columns.duplicated(keep=False).sum())
    if duplicate_column_count:
        frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
    missing_columns = sorted(REQUIRED_CASE_COLUMNS - set(frame.columns))
    counts: dict[str, int] = {
        "empty_dataset_count": int(frame.empty),
        "duplicate_column_count": duplicate_column_count,
        "duplicate_case_count": 0,
        "required_null_count": 0,
        "invalid_identifier_count": 0,
        "invalid_decision_time_count": 0,
        "invalid_category_count": 0,
        "invalid_split_count": 0,
        "missing_split_count": 0,
        "invalid_cause_count": 0,
        "training_missing_cause_count": 0,
        "invalid_numeric_count": 0,
        "invalid_integer_count": 0,
        "non_finite_numeric_count": 0,
        "negative_numeric_count": 0,
        "invalid_binary_count": 0,
        "invalid_compliance_count": 0,
        "invalid_oracle_action_count": 0,
        "invalid_harm_count": 0,
        "invalid_critical_count": 0,
        "entity_split_leakage_count": 0,
    }
    if "case_id" in frame:
        counts["duplicate_case_count"] = int(frame["case_id"].duplicated(keep=False).sum())

    available_required = sorted(REQUIRED_CASE_COLUMNS.intersection(frame.columns))
    non_nullable = [column for column in available_required if column not in NULLABLE_OBSERVABLES]
    if non_nullable:
        counts["required_null_count"] = int(frame[non_nullable].isna().sum().sum())

    for column in ("case_id", "trade_id", "entity_id"):
        if column in frame:
            values = frame[column]
            invalid = values.isna() | values.map(
                lambda value: (
                    not isinstance(value, str) or not value.strip() or value != value.strip()
                )
            )
            counts["invalid_identifier_count"] += int(invalid.sum())
    if "decision_time" in frame:
        counts["invalid_decision_time_count"] = int(
            (~frame["decision_time"].map(_is_timezone_aware)).sum()
        )

    for column, allowed in VALID_CATEGORIES.items():
        if column in frame:
            counts["invalid_category_count"] += int((~frame[column].isin(allowed)).sum())
    if "split" in frame:
        counts["invalid_split_count"] = int((~frame["split"].isin(VALID_SPLITS)).sum())
        counts["missing_split_count"] = len(VALID_SPLITS - set(frame["split"].dropna()))
    if "root_cause" in frame:
        counts["invalid_cause_count"] = int((~frame["root_cause"].isin(CAUSES)).sum())
    if {"split", "root_cause"}.issubset(frame.columns):
        training_causes = set(frame.loc[frame["split"] == "train", "root_cause"].dropna())
        counts["training_missing_cause_count"] = len(set(CAUSES) - training_causes)

    numeric_columns = sorted(
        ({*BASE_NUMERIC_FEATURES, "critical", "harm"}).intersection(frame.columns)
    )
    numeric_values: dict[str, pd.Series] = {}
    for column in numeric_columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        numeric_values[column] = converted
        non_null = frame[column].notna()
        counts["invalid_numeric_count"] += int((non_null & converted.isna()).sum())
        finite = converted.notna()
        counts["non_finite_numeric_count"] += int(
            (finite.to_numpy() & ~np.isfinite(converted.fillna(0.0).to_numpy(dtype=float))).sum()
        )
        if column in NON_NEGATIVE_COLUMNS:
            counts["negative_numeric_count"] += int((converted < 0).sum())

    for column in BINARY_COLUMNS.intersection(numeric_values):
        values = numeric_values[column]
        counts["invalid_binary_count"] += int((values.notna() & ~values.isin([0.0, 1.0])).sum())
    for column in INTEGER_COLUMNS.intersection(numeric_values):
        values = numeric_values[column]
        boolean_values = frame[column].map(lambda value: isinstance(value, (bool, np.bool_)))
        non_integral = values.notna() & (values % 1 != 0)
        counts["invalid_integer_count"] += int((boolean_values | non_integral).sum())
    if "compliance_score" in numeric_values:
        values = numeric_values["compliance_score"]
        counts["invalid_compliance_count"] = int(
            (values.notna() & ((values < 0.0) | (values > 1.0))).sum()
        )

    if {"root_cause", "oracle_action"}.issubset(frame.columns):
        expected_action = frame["root_cause"].map(ACTION_BY_CAUSE)
        counts["invalid_oracle_action_count"] = int(
            (expected_action.notna() & (frame["oracle_action"] != expected_action)).sum()
        )
    if "root_cause" in frame and "harm" in numeric_values:
        expected_harm = frame["root_cause"].map(HARM_BY_CAUSE).astype(float)
        actual_harm = numeric_values["harm"]
        comparable = expected_harm.notna() & actual_harm.notna()
        close = np.isclose(actual_harm.fillna(0.0).to_numpy(), expected_harm.fillna(0.0).to_numpy())
        counts["invalid_harm_count"] = int((comparable.to_numpy() & ~close).sum())
    if "root_cause" in frame and "critical" in numeric_values:
        expected_critical = frame["root_cause"].isin(CRITICAL_CAUSES).astype(float)
        actual_critical = numeric_values["critical"]
        counts["invalid_critical_count"] = int(
            (actual_critical.notna() & (actual_critical != expected_critical)).sum()
        )

    if {"entity_id", "split"}.issubset(frame.columns):
        entity_split_count = (
            frame.dropna(subset=["entity_id", "split"]).groupby("entity_id")["split"].nunique()
        )
        counts["entity_split_leakage_count"] = int((entity_split_count > 1).sum())

    invalid_schema_count = int(sum(counts.values()))
    target_distribution = (
        {
            str(key): int(value)
            for key, value in frame["root_cause"].value_counts().to_dict().items()
        }
        if "root_cause" in frame
        else {}
    )
    split_distribution = (
        {str(key): int(value) for key, value in frame["split"].value_counts().to_dict().items()}
        if "split" in frame
        else {}
    )
    report: dict[str, Any] = {
        "row_count": int(len(frame)),
        "feature_count": max(0, int(len(frame.columns) - 4)),
        "missing_columns": missing_columns,
        **counts,
        "invalid_schema_count": invalid_schema_count,
        "missingness": {
            str(key): float(value) for key, value in frame.isna().mean().to_dict().items()
        },
        "target_distribution": target_distribution,
        "split_distribution": split_distribution,
        "critical_count": int(pd.to_numeric(frame["critical"], errors="coerce").fillna(0).sum())
        if "critical" in frame
        else 0,
        "leakage_check": "PASS" if counts["entity_split_leakage_count"] == 0 else "FAIL",
    }
    report["status"] = "PASS" if not missing_columns and invalid_schema_count == 0 else "FAIL"
    return report
