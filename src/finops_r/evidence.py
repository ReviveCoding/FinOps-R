from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .constants import ACTION_BY_CAUSE


@dataclass(frozen=True)
class RuleEvidence:
    evidence_id: str
    cause: str
    action: str
    strength: float
    reason_code: str


def extract_rule_evidence(row: pd.Series) -> list[RuleEvidence]:
    evidence: list[RuleEvidence] = []

    def add(cause: str, strength: float, code: str) -> None:
        evidence.append(
            RuleEvidence(
                f"{row['case_id']}:{code}", cause, ACTION_BY_CAUSE[cause], float(strength), code
            )
        )

    values = {
        "compliance_score": row.get("compliance_score"),
        "funding_ratio": row.get("funding_ratio"),
        "securities_ratio": row.get("securities_ratio"),
        "ssi_version_gap": row.get("ssi_version_gap"),
        "qty_diff_pct": row.get("qty_diff_pct"),
        "price_diff_bps": row.get("price_diff_bps"),
        "duplicate_amendment_count": row.get("duplicate_amendment_count"),
    }
    if pd.notna(values["compliance_score"]) and float(values["compliance_score"]) >= 0.55:
        add("COMPLIANCE_HOLD", min(1.0, float(values["compliance_score"])), "HIGH_COMPLIANCE_SCORE")
    if pd.notna(values["funding_ratio"]) and float(values["funding_ratio"]) < 0.96:
        add(
            "FUNDING_SHORTAGE",
            min(1.0, 0.70 + 2.0 * max(0.0, 0.92 - float(values["funding_ratio"]))),
            "LOW_FUNDING_RATIO",
        )
    if pd.notna(values["securities_ratio"]) and float(values["securities_ratio"]) < 0.96:
        add(
            "SECURITIES_SHORTAGE",
            min(1.0, 0.70 + 2.0 * max(0.0, 0.92 - float(values["securities_ratio"]))),
            "LOW_SECURITIES_RATIO",
        )
    flag_rules = (
        ("security_id_match", "SECURITY_ID_MISMATCH", "SECURITY_ID_CONFLICT"),
        ("currency_match", "CURRENCY_MISMATCH", "CURRENCY_CONFLICT"),
        ("counterparty_match", "COUNTERPARTY_MISMATCH", "COUNTERPARTY_CONFLICT"),
        ("ssi_present", "MISSING_SSI", "SSI_MISSING"),
    )
    for field, cause, code in flag_rules:
        if row.get(field, 1.0) == 0:
            add(cause, 0.95, code)
    if pd.notna(values["ssi_version_gap"]) and float(values["ssi_version_gap"]) >= 1:
        add(
            "STALE_SSI",
            min(1.0, 0.65 + 0.08 * float(values["ssi_version_gap"])),
            "STALE_SSI_VERSION",
        )
    if pd.notna(values["qty_diff_pct"]) and float(values["qty_diff_pct"]) >= 0.015:
        add(
            "QUANTITY_MISMATCH",
            min(1.0, 0.5 + 5.0 * float(values["qty_diff_pct"])),
            "QUANTITY_DIFFERENCE",
        )
    if pd.notna(values["price_diff_bps"]) and float(values["price_diff_bps"]) >= 15:
        add(
            "PRICE_MISMATCH",
            min(1.0, 0.45 + float(values["price_diff_bps"]) / 120),
            "PRICE_DIFFERENCE",
        )
    if (
        pd.notna(values["duplicate_amendment_count"])
        and float(values["duplicate_amendment_count"]) >= 2
    ):
        add("DUPLICATE_AMENDMENT", 0.92, "DUPLICATE_AMENDMENT_COUNT")
    if row.get("allocation_present", 1.0) == 0 and row.get("state") in {"EXECUTED", "ALLOCATED"}:
        strength = min(1.0, 0.55 + max(0.0, float(row.get("time_in_state_min", 0))) / 300)
        add("LATE_ALLOCATION", strength, "ALLOCATION_MISSING")
    if row.get("affirmation_present", 1.0) == 0 and row.get("state") == "CONFIRMED":
        strength = min(1.0, 0.55 + max(0.0, float(row.get("time_in_state_min", 0))) / 300)
        add("MISSING_AFFIRMATION", strength, "AFFIRMATION_MISSING")
    return evidence


def allowed_actions(row: pd.Series) -> set[str]:
    actions = {
        "NO_ACTION_MONITOR",
        "CORRECT_COUNTERPARTY",
        "CORRECT_ECONOMIC_FIELD",
        "CORRECT_SECURITY_IDENTIFIER",
        "RETRACT_DUPLICATE_AMENDMENT",
        "REQUEST_FUNDING_REVIEW",
        "REQUEST_SECURITIES_REVIEW",
        "ESCALATE_COMPLIANCE_HOLD",
    }
    state = row.get("state")
    if state in {"EXECUTED", "ALLOCATED"}:
        actions.add("REQUEST_ALLOCATION")
    if state in {"ALLOCATED", "CONFIRMED"}:
        actions.add("REQUEST_AFFIRMATION")
    if state in {"INSTRUCTED", "MATCHED"}:
        actions.add("REQUEST_SSI_REFRESH")
    return actions


def build_action_envelopes(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    classes: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    threshold = float(config["action_envelope"]["cumulative_probability"])
    max_candidates = int(config["action_envelope"]["max_candidates"])
    expand_threshold = float(config["action_envelope"]["rule_expand_threshold"])
    records: list[dict[str, Any]] = []
    for row_position, (_, row) in enumerate(frame.iterrows()):
        order = np.argsort(probabilities[row_position])[::-1]
        cause_candidates: list[str] = []
        cumulative = 0.0
        for class_idx in order:
            cause_candidates.append(str(classes[class_idx]))
            cumulative += float(probabilities[row_position, class_idx])
            if cumulative >= threshold or len(cause_candidates) >= max_candidates:
                break
        base_actions = list(dict.fromkeys(ACTION_BY_CAUSE[cause] for cause in cause_candidates))
        evidence = extract_rule_evidence(row)
        expanded_actions = list(base_actions)
        for item in evidence:
            if item.strength >= expand_threshold and item.action not in expanded_actions:
                expanded_actions.append(item.action)
        permitted = allowed_actions(row)
        pruned = [action for action in expanded_actions if action in permitted]
        evidence_actions = {item.action for item in evidence}
        strongest = max(evidence, key=lambda item: item.strength, default=None)
        top_action = base_actions[0]
        conflict_threshold = max(0.90, expand_threshold)
        evidence_conflict = bool(
            strongest
            and strongest.action != top_action
            and strongest.strength >= conflict_threshold
        )
        # Expansion creates a coverage envelope, but conflicting rule evidence does not
        # silently replace the learned point action. It triggers verification and review.
        proposed = (
            top_action if top_action in permitted else (pruned[0] if pruned else "HOLD_FOR_REVIEW")
        )
        support = (
            proposed == "NO_ACTION_MONITOR" and not evidence_conflict
        ) or proposed in evidence_actions
        critical_observables = (
            "ssi_version_gap",
            "qty_diff_pct",
            "price_diff_bps",
            "funding_ratio",
            "securities_ratio",
            "compliance_score",
        )
        false_clear_missingness = proposed == "NO_ACTION_MONITOR" and any(
            pd.isna(row.get(key)) for key in critical_observables
        )
        if float(row.get("evidence_complete", 0.0) or 0.0) < 0.5 or false_clear_missingness:
            verifier = "INSUFFICIENT"
        elif evidence_conflict:
            verifier = "CONTRADICTED"
        elif support:
            verifier = "SUPPORTED"
        else:
            verifier = "INSUFFICIENT"
        records.append(
            {
                "case_id": row["case_id"],
                "cause_candidates": "|".join(cause_candidates),
                "base_actions": "|".join(base_actions),
                "expanded_actions": "|".join(expanded_actions),
                "pruned_actions": "|".join(pruned),
                "proposed_action": proposed,
                "evidence_ids": "|".join(item.evidence_id for item in evidence),
                "evidence_reason_codes": "|".join(item.reason_code for item in evidence),
                "coverage_probability": cumulative,
                "overreach_count": max(0, len(pruned) - 1),
                "policy_status": "PASS" if pruned else "BLOCK",
                "verifier_status": verifier,
                "evidence_supported": int(support and verifier == "SUPPORTED"),
                "evidence_conflict": int(evidence_conflict),
                "false_clear_missingness": int(false_clear_missingness),
            }
        )
    return pd.DataFrame(records, index=frame.index)


def envelope_metrics(envelopes: pd.DataFrame, oracle_actions: pd.Series) -> dict[str, float]:
    recalls: list[float] = []
    precisions: list[float] = []
    sizes: list[int] = []
    for actions_text, oracle in zip(envelopes["pruned_actions"], oracle_actions, strict=True):
        actions = [item for item in str(actions_text).split("|") if item]
        hit = float(str(oracle) in actions)
        recalls.append(hit)
        precisions.append(hit / max(1, len(actions)))
        sizes.append(len(actions))
    recall = float(np.mean(recalls))
    precision = float(np.mean(precisions))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    supported = envelopes["evidence_supported"].to_numpy(dtype=bool)
    action_correct = envelopes["proposed_action"].to_numpy() == oracle_actions.to_numpy()
    supported_precision = float(action_correct[supported].mean()) if supported.any() else 0.0
    return {
        "action_envelope_recall": recall,
        "action_envelope_precision": precision,
        "action_envelope_f1": f1,
        "action_envelope_mean_size": float(np.mean(sizes)),
        "action_envelope_overreach_rate": float(np.mean(np.asarray(sizes) > 1)),
        "evidence_coverage": float(supported.mean()),
        "evidence_precision": supported_precision,
    }
