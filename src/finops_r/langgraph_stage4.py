from __future__ import annotations

from typing import Any, NotRequired, TypedDict, cast

from .orchestration_stage3 import resume_human_review
from .orchestration_stage4 import (
    Stage4Inputs,
    Stage4State,
    build_stage4_state,
    run_stage4_reference,
)


class LangGraphUnavailableError(RuntimeError):
    """Raised when the optional LangGraph runtime is not installed."""


class _GraphState(TypedDict):
    inputs: dict[str, Any]
    result: NotRequired[dict[str, Any]]


def build_langgraph_stage4(checkpointer: Any) -> Any:
    """Build the optional LangGraph path using the Stage 4 reference node contract."""

    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt
    except ImportError as exc:
        raise LangGraphUnavailableError(
            "Install the orchestration extra with: pip install -e '.[orchestration]'"
        ) from exc

    def execute_node(state: _GraphState) -> _GraphState:
        inputs = Stage4Inputs.model_validate(state["inputs"])
        result = run_stage4_reference(inputs)
        return {"inputs": state["inputs"], "result": result.model_dump(mode="json")}

    def route_after_execute(state: _GraphState) -> str:
        result = Stage4State.model_validate(state["result"])
        return "human_review" if result.direct_state.status == "WAITING_REVIEW" else END

    def human_review_node(state: _GraphState) -> _GraphState:
        inputs = Stage4Inputs.model_validate(state["inputs"])
        pending = Stage4State.model_validate(state["result"])
        request = pending.direct_state.review_request
        if request is None:
            raise RuntimeError("review node requires one pending review request")
        response = interrupt(
            {
                "schema_version": "1.0",
                "case_id": request.case_id,
                "action_code": request.action_code,
                "review_token": request.review_token,
                "allowed_decisions": ["APPROVE", "REJECT"],
            }
        )
        if not isinstance(response, dict) or set(response) != {
            "review_token",
            "reviewer_id",
            "decision",
        }:
            raise ValueError("review response schema is incompatible")
        direct = resume_human_review(
            pending.direct_state,
            review_token=response["review_token"],
            reviewer_id=response["reviewer_id"],
            decision=response["decision"],
        )
        result = build_stage4_state(inputs, direct)
        return {"inputs": state["inputs"], "result": result.model_dump(mode="json")}

    builder = StateGraph(_GraphState)
    builder.add_node("stage4_reference", execute_node)
    builder.add_node("human_review", human_review_node)
    builder.add_edge(START, "stage4_reference")
    builder.add_conditional_edges("stage4_reference", route_after_execute)
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=checkpointer)


def invoke_langgraph_stage4(
    graph: Any,
    inputs: Stage4Inputs,
    *,
    thread_id: str,
) -> Stage4State:
    result = cast(
        _GraphState,
        graph.invoke(
            {"inputs": inputs.model_dump(mode="json")},
            config={"configurable": {"thread_id": thread_id}},
        ),
    )
    return Stage4State.model_validate(result["result"])


def resume_langgraph_stage4(
    graph: Any,
    *,
    thread_id: str,
    review_token: str,
    reviewer_id: str,
    decision: str,
) -> Stage4State:
    try:
        from langgraph.types import Command
    except ImportError as exc:
        raise LangGraphUnavailableError(
            "Install the orchestration extra with: pip install -e '.[orchestration]'"
        ) from exc
    result = cast(
        _GraphState,
        graph.invoke(
            Command(
                resume={
                    "review_token": review_token,
                    "reviewer_id": reviewer_id,
                    "decision": decision,
                }
            ),
            config={"configurable": {"thread_id": thread_id}},
        ),
    )
    return Stage4State.model_validate(result["result"])
