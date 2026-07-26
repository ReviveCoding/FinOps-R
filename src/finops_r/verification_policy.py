from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from .planner import ActionPacket, PlannerRequest
from .policy_cards import CompiledPolicyCard
from .retrieval import RetrievalHit


VERIFIER_SCHEMA_VERSION = "1.0"
VERIFIER_ID = "finops_deterministic_policy_verifier_v1"
CLAIM_BOUNDARY = "CONTROLLED_TRADEOPS_SIM_ONLY"
SIMULATION_TARGET = "TRADEOPS_SIM"

AUTO_SIMULATABLE_ACTIONS = frozenset(
    {
        "NO_ACTION_MONITOR",
        "REQUEST_ALLOCATION",
        "REQUEST_AFFIRMATION",
        "REQUEST_SSI_REFRESH",
    }
)
MANDATORY_REVIEW_ACTIONS = frozenset(
    {
        "CORRECT_COUNTERPARTY",
        "CORRECT_ECONOMIC_FIELD",
        "CORRECT_SECURITY_IDENTIFIER",
        "RETRACT_DUPLICATE_AMENDMENT",
        "REQUEST_FUNDING_REVIEW",
        "REQUEST_SECURITIES_REVIEW",
        "ESCALATE_COMPLIANCE_HOLD",
    }
)


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["SUPPORTED", "ABSTAINED", "REJECTED"]
    verifier_id: Literal["finops_deterministic_policy_verifier_v1"] = (
        "finops_deterministic_policy_verifier_v1"
    )
    learned_verifier_used: StrictBool = False
    verified_evidence_ids: tuple[str, ...] = ()
    violation_rule_ids: tuple[str, ...] = ()


class PolicyContext(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    requested_target: str = Field(min_length=1, max_length=80)
    external_execution_requested: StrictBool
    review_capacity_available: StrictBool
    claim_boundary: str = Field(min_length=1, max_length=120)


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["PASS", "HOLD"]
    route: Literal["SIMULATE", "REVIEW", "HOLD"]
    reason_code: Literal[
        "LOW_RISK_SIMULATION",
        "MANDATORY_HUMAN_REVIEW",
        "PLANNER_ABSTENTION",
        "VERIFIER_REJECTED",
        "POLICY_GATE_FAILED",
        "REVIEW_CAPACITY_UNAVAILABLE",
    ]
    violation_rule_ids: tuple[str, ...] = ()
    external_execution: StrictBool = False
    target: Literal["TRADEOPS_SIM"] = "TRADEOPS_SIM"


def verify_action_packet(
    packet: ActionPacket,
    request: PlannerRequest,
    hits: list[RetrievalHit],
    cards: list[CompiledPolicyCard],
) -> VerificationResult:
    """Independently verify one planner packet against retrieved, effective policy cards."""

    violations: list[str] = []
    if packet.case_id != request.case_id:
        violations.append("VERIFIER_CASE_ID_MISMATCH")
    if packet.decision_time != request.decision_time:
        violations.append("VERIFIER_DECISION_TIME_MISMATCH")
    if packet.root_cause != request.predicted_cause:
        violations.append("VERIFIER_ROOT_CAUSE_MISMATCH")

    holding = packet.action_code == "HOLD_FOR_REVIEW"
    if holding and packet.evidence_ids:
        violations.append("VERIFIER_HOLD_MUST_NOT_CITE_EVIDENCE")
    if not holding and packet.action_code not in request.permitted_actions:
        violations.append("VERIFIER_ACTION_NOT_PERMITTED")

    hits_by_id = {hit.document_id: hit for hit in hits}
    cards_by_id = {card.card_id: card for card in cards}
    if len(hits_by_id) != len(hits):
        violations.append("VERIFIER_DUPLICATE_RETRIEVAL_ID")
    if len(cards_by_id) != len(cards):
        violations.append("VERIFIER_DUPLICATE_POLICY_CARD_ID")

    verified: list[str] = []
    for evidence_id in packet.evidence_ids:
        hit = hits_by_id.get(evidence_id)
        if hit is None:
            violations.append("VERIFIER_UNRETRIEVED_EVIDENCE")
            continue
        if hit.source_type != "POLICY_CARD":
            violations.append("VERIFIER_NON_POLICY_EVIDENCE")
            continue
        card = cards_by_id.get(evidence_id)
        if card is None:
            violations.append("VERIFIER_POLICY_CARD_NOT_FOUND")
            continue
        try:
            card.verify_checksum()
        except ValueError:
            violations.append("VERIFIER_POLICY_CARD_CHECKSUM_MISMATCH")
            continue
        if hit.checksum != card.checksum:
            violations.append("VERIFIER_RETRIEVAL_CARD_CHECKSUM_MISMATCH")
            continue
        if not card.is_effective_at(request.decision_time):
            violations.append("VERIFIER_POLICY_CARD_NOT_EFFECTIVE")
            continue
        if not card.legal_interpretation_prohibited:
            violations.append("VERIFIER_LEGAL_BOUNDARY_DISABLED")
            continue
        if (
            packet.action_code not in hit.allowed_actions
            or packet.action_code not in card.allowed_actions
        ):
            violations.append("VERIFIER_CITATION_DOES_NOT_SUPPORT_ACTION")
            continue
        verified.append(evidence_id)

    if not holding and not packet.evidence_ids:
        violations.append("VERIFIER_MISSING_ACTION_CITATION")
    if not holding and len(verified) != len(packet.evidence_ids):
        violations.append("VERIFIER_INCOMPLETE_CITATION_SUPPORT")

    unique_violations = tuple(sorted(set(violations)))
    if unique_violations:
        return VerificationResult(
            status="REJECTED",
            verified_evidence_ids=tuple(verified),
            violation_rule_ids=unique_violations,
        )
    if holding:
        return VerificationResult(status="ABSTAINED")
    return VerificationResult(
        status="SUPPORTED",
        verified_evidence_ids=tuple(verified),
    )


def apply_simulation_policy(
    packet: ActionPacket,
    verification: VerificationResult,
    context: PolicyContext,
) -> PolicyDecision:
    """Apply the independent simulation-only policy gate and fail closed."""

    boundary_violations: list[str] = []
    if context.claim_boundary != CLAIM_BOUNDARY:
        boundary_violations.append("POLICY_CLAIM_BOUNDARY_MISMATCH")
    if context.requested_target != SIMULATION_TARGET:
        boundary_violations.append("POLICY_EXTERNAL_TARGET_PROHIBITED")
    if context.external_execution_requested:
        boundary_violations.append("POLICY_EXTERNAL_EXECUTION_PROHIBITED")
    if boundary_violations:
        return PolicyDecision(
            status="HOLD",
            route="HOLD",
            reason_code="POLICY_GATE_FAILED",
            violation_rule_ids=tuple(sorted(boundary_violations)),
        )
    if verification.status == "REJECTED":
        return PolicyDecision(
            status="HOLD",
            route="HOLD",
            reason_code="VERIFIER_REJECTED",
            violation_rule_ids=verification.violation_rule_ids,
        )
    if packet.action_code == "HOLD_FOR_REVIEW" or verification.status == "ABSTAINED":
        return PolicyDecision(
            status="HOLD",
            route="HOLD",
            reason_code="PLANNER_ABSTENTION",
        )
    if packet.action_code in MANDATORY_REVIEW_ACTIONS:
        if not context.review_capacity_available:
            return PolicyDecision(
                status="HOLD",
                route="HOLD",
                reason_code="REVIEW_CAPACITY_UNAVAILABLE",
                violation_rule_ids=("POLICY_REVIEW_CAPACITY_UNAVAILABLE",),
            )
        return PolicyDecision(
            status="PASS",
            route="REVIEW",
            reason_code="MANDATORY_HUMAN_REVIEW",
        )
    if packet.action_code in AUTO_SIMULATABLE_ACTIONS:
        return PolicyDecision(
            status="PASS",
            route="SIMULATE",
            reason_code="LOW_RISK_SIMULATION",
        )
    return PolicyDecision(
        status="HOLD",
        route="HOLD",
        reason_code="POLICY_GATE_FAILED",
        violation_rule_ids=("POLICY_ACTION_CLASS_UNDEFINED",),
    )
