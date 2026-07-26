from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from .json_safety import strict_json_dumps
from .planner import ActionPacket, PlannerRequest
from .policy_cards import CompiledPolicyCard
from .retrieval import RetrievalHit
from .verification_policy import (
    PolicyContext,
    PolicyDecision,
    VerificationResult,
    apply_simulation_policy,
    verify_action_packet,
)


ORCHESTRATION_SCHEMA_VERSION = "1.0"
TRACE_NODES = frozenset({"INGEST", "VERIFY", "POLICY", "ROUTE", "REVIEW", "SIMULATE", "COMPLETE"})


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    payload = strict_json_dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    sequence: int = Field(ge=1)
    node: Literal["INGEST", "VERIFY", "POLICY", "ROUTE", "REVIEW", "SIMULATE", "COMPLETE"]
    outcome: str = Field(min_length=1, max_length=120)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SimulationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    case_id: str
    action_code: str
    target: Literal["TRADEOPS_SIM"] = "TRADEOPS_SIM"
    external_execution: StrictBool = False
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["SIMULATED"] = "SIMULATED"


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    case_id: str
    action_code: str
    review_token: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["PENDING"] = "PENDING"


class ReviewResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    reviewer_id: str = Field(min_length=1, max_length=160)
    decision: Literal["APPROVE", "REJECT"]


class OrchestrationState(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    case_id: str
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["COMPLETED", "WAITING_REVIEW"]
    route: Literal["SIMULATE", "REVIEW", "HOLD"]
    verification: VerificationResult
    policy: PolicyDecision
    trace: tuple[TraceEvent, ...]
    receipt: SimulationReceipt | None = None
    review_request: ReviewRequest | None = None
    review_resolution: ReviewResolution | None = None
    trace_complete: StrictBool


def _event(
    trace: list[TraceEvent],
    node: Literal["INGEST", "VERIFY", "POLICY", "ROUTE", "REVIEW", "SIMULATE", "COMPLETE"],
    outcome: str,
    input_value: Any,
    output_value: Any,
) -> None:
    trace.append(
        TraceEvent(
            sequence=len(trace) + 1,
            node=node,
            outcome=outcome,
            input_sha256=_fingerprint(input_value),
            output_sha256=_fingerprint(output_value),
        )
    )


def _trace_is_complete(
    trace: tuple[TraceEvent, ...] | list[TraceEvent],
    *,
    status: Literal["COMPLETED", "WAITING_REVIEW"],
    route: Literal["SIMULATE", "REVIEW", "HOLD"],
) -> bool:
    if [event.sequence for event in trace] != list(range(1, len(trace) + 1)):
        return False
    if any(event.node not in TRACE_NODES for event in trace):
        return False
    nodes = [event.node for event in trace]
    required = {"INGEST", "VERIFY", "POLICY", "ROUTE"}
    if status == "WAITING_REVIEW":
        return required.issubset(nodes) and route == "REVIEW" and nodes[-1] == "ROUTE"
    required.add("COMPLETE")
    if route == "SIMULATE":
        required.add("SIMULATE")
    if any(event.node == "REVIEW" for event in trace):
        required.add("REVIEW")
    return required.issubset(nodes) and nodes[-1] == "COMPLETE"


def trace_contract_is_complete(state: OrchestrationState) -> bool:
    """Independently recompute the route-specific trace completeness contract."""

    return state.trace_complete and _trace_is_complete(
        state.trace,
        status=state.status,
        route=state.route,
    )


def _idempotency_key(request: PlannerRequest, packet: ActionPacket) -> str:
    return _fingerprint(
        {
            "contract": "FinOps-R/direct-orchestrator/1.0",
            "request": request.model_dump(mode="json"),
            "packet": packet.model_dump(mode="json"),
        }
    )


def run_direct_orchestrator(
    request: PlannerRequest,
    packet: ActionPacket,
    hits: list[RetrievalHit],
    cards: list[CompiledPolicyCard],
    context: PolicyContext,
) -> OrchestrationState:
    """Run the typed direct pipeline before any graph-framework adapter is introduced."""

    trace: list[TraceEvent] = []
    key = _idempotency_key(request, packet)
    _event(trace, "INGEST", "BOUND", request, {"packet": packet, "idempotency_key": key})

    verification = verify_action_packet(packet, request, hits, cards)
    _event(trace, "VERIFY", verification.status, packet, verification)
    policy = apply_simulation_policy(packet, verification, context)
    _event(
        trace,
        "POLICY",
        policy.status,
        {"verification": verification, "context": context},
        policy,
    )
    _event(trace, "ROUTE", policy.route, policy, {"route": policy.route})

    receipt: SimulationReceipt | None = None
    review_request: ReviewRequest | None = None
    if policy.route == "SIMULATE":
        receipt = SimulationReceipt(
            case_id=request.case_id,
            action_code=packet.action_code,
            idempotency_key=key,
        )
        _event(trace, "SIMULATE", "SIMULATED", packet, receipt)
        _event(trace, "COMPLETE", "COMPLETED", receipt, {"status": "COMPLETED"})
        status: Literal["COMPLETED", "WAITING_REVIEW"] = "COMPLETED"
    elif policy.route == "REVIEW":
        review_request = ReviewRequest(
            case_id=request.case_id,
            action_code=packet.action_code,
            review_token=_fingerprint({"idempotency_key": key, "purpose": "HUMAN_REVIEW"}),
        )
        status = "WAITING_REVIEW"
    else:
        _event(trace, "COMPLETE", "HELD", policy, {"status": "COMPLETED"})
        status = "COMPLETED"

    complete = _trace_is_complete(trace, status=status, route=policy.route)
    if not complete:
        raise RuntimeError("orchestration trace contract is incomplete")
    return OrchestrationState(
        case_id=request.case_id,
        idempotency_key=key,
        status=status,
        route=policy.route,
        verification=verification,
        policy=policy,
        trace=tuple(trace),
        receipt=receipt,
        review_request=review_request,
        trace_complete=True,
    )


def resume_human_review(
    state: OrchestrationState,
    *,
    review_token: str,
    reviewer_id: str,
    decision: Literal["APPROVE", "REJECT"],
) -> OrchestrationState:
    """Resume one pending review exactly once, while allowing byte-identical replay."""

    resolution = ReviewResolution(reviewer_id=reviewer_id, decision=decision)
    if state.review_resolution is not None:
        if (
            state.review_resolution == resolution
            and state.review_request is not None
            and state.review_request.review_token == review_token
        ):
            return state
        raise ValueError("review has already been resolved with different inputs")
    if state.status != "WAITING_REVIEW" or state.route != "REVIEW":
        raise ValueError("orchestration state is not waiting for review")
    if state.review_request is None or state.review_request.review_token != review_token:
        raise ValueError("review token is invalid")

    trace = list(state.trace)
    _event(trace, "REVIEW", decision, state.review_request, resolution)
    receipt: SimulationReceipt | None = None
    if decision == "APPROVE":
        receipt = SimulationReceipt(
            case_id=state.case_id,
            action_code=state.review_request.action_code,
            idempotency_key=state.idempotency_key,
        )
        _event(trace, "SIMULATE", "SIMULATED_AFTER_REVIEW", resolution, receipt)
        route: Literal["SIMULATE", "HOLD"] = "SIMULATE"
        final_input: Any = receipt
        outcome = "COMPLETED"
    else:
        route = "HOLD"
        final_input = resolution
        outcome = "REVIEW_REJECTED"
    _event(trace, "COMPLETE", outcome, final_input, {"status": "COMPLETED", "route": route})
    complete = _trace_is_complete(trace, status="COMPLETED", route=route)
    if not complete:
        raise RuntimeError("resumed orchestration trace contract is incomplete")
    return OrchestrationState(
        case_id=state.case_id,
        idempotency_key=state.idempotency_key,
        status="COMPLETED",
        route=route,
        verification=state.verification,
        policy=state.policy,
        trace=tuple(trace),
        receipt=receipt,
        review_request=state.review_request,
        review_resolution=resolution,
        trace_complete=True,
    )
