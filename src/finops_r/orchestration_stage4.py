from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .checkpointing_stage4 import CheckpointConflictError, SQLiteCheckpointStore
from .json_safety import strict_json_dumps
from .orchestration_stage3 import (
    OrchestrationState,
    ReviewResolution,
    SimulationReceipt,
    resume_human_review,
    run_direct_orchestrator,
)
from .planner import ActionPacket, PlannerRequest
from .policy_cards import CompiledPolicyCard
from .retrieval import RetrievalHit
from .verification_policy import PolicyContext
from .verifier_stage4 import (
    FrozenVerifierArtifact,
    LearnedVerifierResult,
    run_frozen_verifier,
)

STAGE4_SCHEMA_VERSION = "1.0"


class InjectedCrash(RuntimeError):
    """Deterministic fault used to qualify checkpoint recovery."""


class Stage4Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    request: PlannerRequest
    packet: ActionPacket
    hits: tuple[RetrievalHit, ...]
    cards: tuple[CompiledPolicyCard, ...]
    context: PolicyContext
    verifier_artifact: FrozenVerifierArtifact
    verifier_artifact_sha256: str


class Stage4State(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    direct_state: OrchestrationState
    learned_verifier: LearnedVerifierResult
    effective_route: Literal["SIMULATE", "REVIEW", "HOLD"]
    simulation_effect_recorded: bool


def _jsonable(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _sha256(value: BaseModel | dict[str, Any]) -> str:
    return hashlib.sha256(
        strict_json_dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_stage4_state(inputs: Stage4Inputs, direct_state: OrchestrationState) -> Stage4State:
    learned = run_frozen_verifier(
        inputs.packet,
        direct_state.verification,
        list(inputs.hits),
        artifact=inputs.verifier_artifact,
        artifact_sha256=inputs.verifier_artifact_sha256,
    )
    effective_route: Literal["SIMULATE", "REVIEW", "HOLD"]
    effective_route = "HOLD" if learned.flagged else direct_state.route
    return Stage4State(
        direct_state=direct_state,
        learned_verifier=learned,
        effective_route=effective_route,
        simulation_effect_recorded=False,
    )


def run_stage4_reference(inputs: Stage4Inputs) -> Stage4State:
    direct = run_direct_orchestrator(
        inputs.request,
        inputs.packet,
        list(inputs.hits),
        list(inputs.cards),
        inputs.context,
    )
    return build_stage4_state(inputs, direct)


class DurableStage4Orchestrator:
    """Framework-independent Stage 4 reference with crash-safe replay."""

    def __init__(self, store: SQLiteCheckpointStore):
        self.store = store

    @staticmethod
    def _recorded(state: Stage4State) -> Stage4State:
        return state.model_copy(update={"simulation_effect_recorded": True})

    def _record_effect(
        self,
        state: Stage4State,
        *,
        crash_at: str | None,
    ) -> Stage4State:
        receipt = state.direct_state.receipt
        if state.effective_route != "SIMULATE" or receipt is None:
            return state
        stored = self.store.record_simulation_once(
            idempotency_key=state.direct_state.idempotency_key,
            request={
                "case_id": state.direct_state.case_id,
                "action_code": receipt.action_code,
                "target": receipt.target,
                "external_execution": receipt.external_execution,
            },
            receipt_factory=lambda: receipt.model_dump(mode="json"),
        )
        if SimulationReceipt.model_validate(stored) != receipt:
            raise CheckpointConflictError("simulation ledger receipt conflicts with state")
        if crash_at == "after_effect":
            raise InjectedCrash("injected crash after idempotent effect commit")
        return self._recorded(state)

    def start(
        self,
        thread_id: str,
        inputs: Stage4Inputs,
        *,
        crash_at: str | None = None,
    ) -> Stage4State:
        input_sha = _sha256(inputs)
        self.store.bind_thread(thread_id, input_sha)
        checkpoints = self.store.load(thread_id)
        if checkpoints:
            if checkpoints[0].kind != "INPUT":
                raise CheckpointConflictError("thread does not begin with an input checkpoint")
            persisted_inputs = Stage4Inputs.model_validate(checkpoints[0].payload)
            if _sha256(persisted_inputs) != input_sha:
                raise CheckpointConflictError("persisted inputs do not match replay inputs")
            if checkpoints[-1].kind == "STATE":
                return Stage4State.model_validate(checkpoints[-1].payload)
        else:
            self.store.append(thread_id, "INPUT", inputs.model_dump(mode="json"))
        if crash_at == "after_input":
            raise InjectedCrash("injected crash after input checkpoint")
        direct = run_direct_orchestrator(
            inputs.request,
            inputs.packet,
            list(inputs.hits),
            list(inputs.cards),
            inputs.context,
        )
        state = build_stage4_state(inputs, direct)
        state = self._record_effect(state, crash_at=crash_at)
        kind = "WAITING_REVIEW" if direct.status == "WAITING_REVIEW" else "STATE"
        self.store.append(thread_id, kind, state.model_dump(mode="json"))
        return state

    def resume_review(
        self,
        thread_id: str,
        *,
        review_token: str,
        reviewer_id: str,
        decision: Literal["APPROVE", "REJECT"],
        crash_at: str | None = None,
    ) -> Stage4State:
        checkpoints = self.store.load(thread_id)
        if not checkpoints or checkpoints[0].kind != "INPUT":
            raise CheckpointConflictError("thread has no durable input checkpoint")
        inputs = Stage4Inputs.model_validate(checkpoints[0].payload)
        latest = checkpoints[-1]
        resolution = ReviewResolution(reviewer_id=reviewer_id, decision=decision)
        review_payload = {
            "review_token": review_token,
            "resolution": resolution.model_dump(mode="json"),
        }
        if latest.kind == "STATE":
            completed = Stage4State.model_validate(latest.payload)
            actual = completed.direct_state.review_resolution
            if (
                actual == resolution
                and completed.direct_state.review_request is not None
                and completed.direct_state.review_request.review_token == review_token
            ):
                return completed
            raise CheckpointConflictError("review was already resolved with different inputs")
        if latest.kind == "REVIEW_INPUT":
            if latest.payload != review_payload:
                raise CheckpointConflictError("durable review input conflicts with replay")
            pending = Stage4State.model_validate(checkpoints[-2].payload)
        elif latest.kind == "WAITING_REVIEW":
            pending = Stage4State.model_validate(latest.payload)
            self.store.append(thread_id, "REVIEW_INPUT", review_payload)
        else:
            raise CheckpointConflictError("thread is not waiting for review")
        direct = resume_human_review(
            pending.direct_state,
            review_token=review_token,
            reviewer_id=reviewer_id,
            decision=decision,
        )
        state = build_stage4_state(inputs, direct)
        state = self._record_effect(state, crash_at=crash_at)
        self.store.append(thread_id, "STATE", state.model_dump(mode="json"))
        return state

    def recover(self, thread_id: str) -> Stage4State:
        checkpoints = self.store.load(thread_id)
        if not checkpoints or checkpoints[0].kind != "INPUT":
            raise CheckpointConflictError("thread has no durable input checkpoint")
        inputs = Stage4Inputs.model_validate(checkpoints[0].payload)
        latest = checkpoints[-1]
        if latest.kind == "STATE":
            return Stage4State.model_validate(latest.payload)
        if latest.kind == "WAITING_REVIEW":
            return Stage4State.model_validate(latest.payload)
        if latest.kind == "REVIEW_INPUT":
            resolution = ReviewResolution.model_validate(latest.payload["resolution"])
            return self.resume_review(
                thread_id,
                review_token=str(latest.payload["review_token"]),
                reviewer_id=resolution.reviewer_id,
                decision=resolution.decision,
            )
        return self.start(thread_id, inputs)

    def release(self, thread_id: str) -> dict[str, Any]:
        state = self.recover(thread_id)
        if state.direct_state.status != "COMPLETED":
            raise CheckpointConflictError("a waiting review cannot be released")
        return self.store.release(thread_id, _sha256(state))


def default_stage4_inputs(
    *,
    request: PlannerRequest,
    packet: ActionPacket,
    hits: list[RetrievalHit],
    cards: list[CompiledPolicyCard],
    context: PolicyContext,
    verifier_path: Path,
) -> Stage4Inputs:
    from .verifier_stage4 import load_frozen_verifier

    artifact, checksum = load_frozen_verifier(verifier_path)
    return Stage4Inputs(
        request=request,
        packet=packet,
        hits=tuple(hits),
        cards=tuple(cards),
        context=context,
        verifier_artifact=artifact,
        verifier_artifact_sha256=checksum,
    )
