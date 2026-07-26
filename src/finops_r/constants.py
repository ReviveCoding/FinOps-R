from __future__ import annotations

STATES = ("EXECUTED", "ALLOCATED", "CONFIRMED", "INSTRUCTED", "MATCHED")
STATE_RANK = {state: idx / (len(STATES) - 1) for idx, state in enumerate(STATES)}

CAUSES = (
    "NO_EXCEPTION",
    "STALE_SSI",
    "MISSING_SSI",
    "COUNTERPARTY_MISMATCH",
    "QUANTITY_MISMATCH",
    "PRICE_MISMATCH",
    "CURRENCY_MISMATCH",
    "LATE_ALLOCATION",
    "MISSING_AFFIRMATION",
    "SECURITY_ID_MISMATCH",
    "DUPLICATE_AMENDMENT",
    "FUNDING_SHORTAGE",
    "SECURITIES_SHORTAGE",
    "COMPLIANCE_HOLD",
)

ACTION_BY_CAUSE = {
    "NO_EXCEPTION": "NO_ACTION_MONITOR",
    "STALE_SSI": "REQUEST_SSI_REFRESH",
    "MISSING_SSI": "REQUEST_SSI_REFRESH",
    "COUNTERPARTY_MISMATCH": "CORRECT_COUNTERPARTY",
    "QUANTITY_MISMATCH": "CORRECT_ECONOMIC_FIELD",
    "PRICE_MISMATCH": "CORRECT_ECONOMIC_FIELD",
    "CURRENCY_MISMATCH": "CORRECT_ECONOMIC_FIELD",
    "LATE_ALLOCATION": "REQUEST_ALLOCATION",
    "MISSING_AFFIRMATION": "REQUEST_AFFIRMATION",
    "SECURITY_ID_MISMATCH": "CORRECT_SECURITY_IDENTIFIER",
    "DUPLICATE_AMENDMENT": "RETRACT_DUPLICATE_AMENDMENT",
    "FUNDING_SHORTAGE": "REQUEST_FUNDING_REVIEW",
    "SECURITIES_SHORTAGE": "REQUEST_SECURITIES_REVIEW",
    "COMPLIANCE_HOLD": "ESCALATE_COMPLIANCE_HOLD",
}

CRITICAL_CAUSES = {
    "COMPLIANCE_HOLD",
    "FUNDING_SHORTAGE",
    "SECURITIES_SHORTAGE",
    "SECURITY_ID_MISMATCH",
}

HARM_BY_CAUSE = {
    "NO_EXCEPTION": 1.0,
    "STALE_SSI": 5.0,
    "MISSING_SSI": 6.0,
    "COUNTERPARTY_MISMATCH": 7.0,
    "QUANTITY_MISMATCH": 6.0,
    "PRICE_MISMATCH": 6.0,
    "CURRENCY_MISMATCH": 7.0,
    "LATE_ALLOCATION": 4.0,
    "MISSING_AFFIRMATION": 5.0,
    "SECURITY_ID_MISMATCH": 9.0,
    "DUPLICATE_AMENDMENT": 5.0,
    "FUNDING_SHORTAGE": 10.0,
    "SECURITIES_SHORTAGE": 10.0,
    "COMPLIANCE_HOLD": 12.0,
}

BASE_NUMERIC_FEATURES = (
    "time_in_state_min",
    "case_age_min",
    "deadline_slack_min",
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
    "allocation_present",
    "affirmation_present",
    "ssi_present",
    "currency_match",
    "counterparty_match",
    "security_id_match",
    "evidence_complete",
)

CATEGORICAL_FEATURES = ("state", "product", "route", "currency", "side")
RSOP_FEATURES = (
    "state_rank",
    "rsop_percentile",
    "delay_pressure",
    "path_divergence",
    "reference_backoff_level",
)
