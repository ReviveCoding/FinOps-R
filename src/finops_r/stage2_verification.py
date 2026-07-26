from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import sha256_file, verify_artifact_manifest
from .json_safety import (
    require_json_object,
    strict_json_load,
    strict_json_loads,
)
from .planner_stage2 import (
    DEFAULT_DIAGNOSTIC_SEEDS,
    DEFAULT_EVALUATION_SEEDS,
    DEVELOPMENT_SCENARIOS_SHA256,
    NON_THINKING_MAX_NEW_TOKENS,
    QUALIFICATION_HOLDOUT_SHA256,
    THINKING_MAX_NEW_TOKENS,
)
from .retrieval_stage2 import (
    BGE_MODEL_REVISION,
    QWEN_MODEL_REVISION,
    STAGE2_BENCHMARK_SHA256,
)


EXPECTED_FILES = {
    "evaluation_input_snapshot.json",
    "planner_mode_metrics.csv",
    "planner_scenario_results.json",
    "policy_card_manifest.json",
    "policy_cards.json",
    "retrieval_candidate_comparison.csv",
    "retrieval_query_results.json",
    "runtime_manifest.json",
    "stage2_summary.json",
}
STAGE2_IMPLEMENTATION_REVISION = "retrieval_planner_stage2_v5"
PRODUCTION_CANDIDATE_MODE = "NON_THINKING"
DIAGNOSTIC_MODE = "THINKING"
PLANNER_QUALITY_GATES = {
    "schema_valid_rate": 1.0,
    "raw_model_citation_precision": 1.0,
    "raw_model_abstention_accuracy": 0.75,
    "raw_model_action_accuracy": 0.75,
    "per_seed_min_raw_model_abstention_accuracy": 0.75,
    "per_seed_min_raw_model_action_accuracy": 0.75,
}
EXPECTED_UNIQUE_SCENARIOS = 12
EXPECTED_SEEDS_BY_MODE = {
    PRODUCTION_CANDIDATE_MODE: tuple(DEFAULT_EVALUATION_SEEDS),
    DIAGNOSTIC_MODE: tuple(DEFAULT_DIAGNOSTIC_SEEDS),
}
EXPECTED_RUNS_BY_MODE = {
    mode: EXPECTED_UNIQUE_SCENARIOS * len(seeds) for mode, seeds in EXPECTED_SEEDS_BY_MODE.items()
}


def _finite_rate(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _close(left: Any, right: Any) -> bool:
    return (
        _finite_rate(left)
        and _finite_rate(right)
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    )


def _text_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def planner_quality_failures(planner_metrics: pd.DataFrame) -> list[str]:
    rows = planner_metrics.loc[
        planner_metrics["mode"].astype(str) == PRODUCTION_CANDIDATE_MODE
    ].to_dict(orient="records")
    if len(rows) != 1:
        return ["PLANNER_QUALITY_PRODUCTION_MODE_ROW"]
    row = rows[0]
    failures: list[str] = []
    for metric, minimum in PLANNER_QUALITY_GATES.items():
        value = row.get(metric)
        if not _finite_rate(value) or float(value) < minimum:
            failures.append(f"PLANNER_QUALITY_{PRODUCTION_CANDIDATE_MODE}_{metric.upper()}")
    return failures


def _recomputed_mode_metrics(records: list[dict[str, Any]], mode: str) -> dict[str, float | int]:
    selected = [item for item in records if item.get("mode") == mode]
    if not selected:
        raise ValueError(f"missing planner records for mode {mode}")
    raw_citation_count = sum(int(item.get("raw_citation_count", -1)) for item in selected)
    raw_supported_count = sum(
        int(item.get("raw_supported_citation_count", -1)) for item in selected
    )
    enforced_citation_count = sum(
        int(item.get("safety_enforced_citation_count", -1)) for item in selected
    )
    enforced_supported_count = sum(
        int(item.get("safety_enforced_supported_citation_count", -1)) for item in selected
    )
    expected_seeds = EXPECTED_SEEDS_BY_MODE[mode]
    by_seed = {
        seed: [item for item in selected if item.get("seed") == seed] for seed in expected_seeds
    }
    return {
        "schema_valid_rate": sum(item.get("schema_valid") is True for item in selected)
        / len(selected),
        "raw_model_citation_precision": (
            1.0 if raw_citation_count == 0 else raw_supported_count / raw_citation_count
        ),
        "safety_enforced_citation_precision": (
            1.0
            if enforced_citation_count == 0
            else enforced_supported_count / enforced_citation_count
        ),
        "raw_model_abstention_accuracy": sum(
            item.get("raw_abstention_correct") is True for item in selected
        )
        / len(selected),
        "raw_model_action_accuracy": sum(
            item.get("raw_action_correct") is True for item in selected
        )
        / len(selected),
        "per_seed_min_raw_model_abstention_accuracy": min(
            sum(item.get("raw_abstention_correct") is True for item in seed_records)
            / len(seed_records)
            for seed_records in by_seed.values()
        ),
        "per_seed_min_raw_model_action_accuracy": min(
            sum(item.get("raw_action_correct") is True for item in seed_records) / len(seed_records)
            for seed_records in by_seed.values()
        ),
        "safety_enforced_abstention_accuracy": sum(
            item.get("safety_abstention_correct") is True for item in selected
        )
        / len(selected),
        "safety_enforced_action_accuracy": sum(
            item.get("safety_action_correct") is True for item in selected
        )
        / len(selected),
        "hit_token_limit_rate": sum(item.get("hit_token_limit") is True for item in selected)
        / len(selected),
        "post_gate_unauthorized_action_count": sum(
            item.get("post_gate_unauthorized") is True for item in selected
        ),
    }


def verify_stage2_artifact(root: Path) -> dict[str, Any]:
    root = root.resolve()
    integrity = verify_artifact_manifest(root)
    manifest_records = strict_json_load(root / "artifact_manifest.json")
    actual_files = {item["path"] for item in manifest_records}
    failures: list[str] = []
    if actual_files != EXPECTED_FILES:
        failures.append("ARTIFACT_FILE_SET")

    runtime = require_json_object(
        strict_json_load(root / "runtime_manifest.json"),
        source="stage-2 runtime manifest",
    )
    if runtime.get("implementation_revision") != STAGE2_IMPLEMENTATION_REVISION:
        failures.append("IMPLEMENTATION_REVISION")
    if runtime.get("benchmark_sha256") != STAGE2_BENCHMARK_SHA256:
        failures.append("BENCHMARK_SHA256")
    if runtime.get("development_scenarios_sha256") != DEVELOPMENT_SCENARIOS_SHA256:
        failures.append("DEVELOPMENT_SCENARIOS_SHA256")
    if runtime.get("qualification_holdout_sha256") != QUALIFICATION_HOLDOUT_SHA256:
        failures.append("QUALIFICATION_HOLDOUT_SHA256")
    models = runtime.get("models", {})
    if models.get("bge_revision") != BGE_MODEL_REVISION:
        failures.append("BGE_REVISION")
    if models.get("qwen_revision") != QWEN_MODEL_REVISION:
        failures.append("QWEN_REVISION")
    generation = runtime.get("generation", {})
    if generation.get("production_evaluation_seeds") != list(
        EXPECTED_SEEDS_BY_MODE[PRODUCTION_CANDIDATE_MODE]
    ):
        failures.append("PRODUCTION_EVALUATION_SEEDS")
    if generation.get("diagnostic_evaluation_seeds") != list(
        EXPECTED_SEEDS_BY_MODE[DIAGNOSTIC_MODE]
    ):
        failures.append("DIAGNOSTIC_EVALUATION_SEEDS")
    thinking = generation.get("thinking", {})
    non_thinking = generation.get("non_thinking", {})
    if (
        thinking.get("max_new_tokens") != THINKING_MAX_NEW_TOKENS
        or thinking.get("role") != "DIAGNOSTIC_ONLY"
    ):
        failures.append("THINKING_GENERATION_CONTRACT")
    if (
        non_thinking.get("max_new_tokens") != NON_THINKING_MAX_NEW_TOKENS
        or non_thinking.get("role") != "PRODUCTION_CANDIDATE"
    ):
        failures.append("NON_THINKING_GENERATION_CONTRACT")
    if runtime.get("external_execution") is not False:
        failures.append("EXTERNAL_EXECUTION_BOUNDARY")
    if runtime.get("chain_of_thought_persisted") is not False:
        failures.append("CHAIN_OF_THOUGHT_BOUNDARY")

    candidates = pd.read_csv(root / "retrieval_candidate_comparison.csv")
    if candidates["method"].tolist() != ["BM25", "BGE_DENSE", "RRF_HYBRID"]:
        failures.append("RETRIEVAL_METHODS")
    if candidates.shape[0] != 3 or set(candidates["query_count"].tolist()) != {35}:
        failures.append("RETRIEVAL_QUERY_COUNT")
    if int(candidates["future_evidence_count"].sum()) != 0:
        failures.append("RETRIEVAL_TEMPORAL_LEAKAGE")
    for column in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5"):
        if not all(_finite_rate(value) for value in candidates[column].tolist()):
            failures.append(f"RETRIEVAL_METRIC_{column.upper()}")

    query_results = require_json_object(
        strict_json_load(root / "retrieval_query_results.json"),
        source="stage-2 retrieval results",
    )
    if query_results.get("query_count") != 35 or len(query_results.get("records", [])) != 105:
        failures.append("RETRIEVAL_RESULT_COUNT")
    if query_results.get("future_evidence_count") != 0:
        failures.append("QUERY_RESULT_TEMPORAL_LEAKAGE")

    snapshot = require_json_object(
        strict_json_load(root / "evaluation_input_snapshot.json"),
        source="stage-2 evaluation input snapshot",
    )
    if snapshot.get("retrieval_benchmark_sha256") != STAGE2_BENCHMARK_SHA256:
        failures.append("SNAPSHOT_RETRIEVAL_SHA256")
    if snapshot.get("development_scenarios_sha256") != DEVELOPMENT_SCENARIOS_SHA256:
        failures.append("SNAPSHOT_DEVELOPMENT_SHA256")
    if snapshot.get("qualification_holdout_sha256") != QUALIFICATION_HOLDOUT_SHA256:
        failures.append("SNAPSHOT_HOLDOUT_SHA256")
    source_text_contract = {
        "retrieval_benchmark_source_text": STAGE2_BENCHMARK_SHA256,
        "development_scenarios_source_text": DEVELOPMENT_SCENARIOS_SHA256,
        "qualification_holdout_source_text": QUALIFICATION_HOLDOUT_SHA256,
        "source_cards_source_text": runtime.get("source_cards_sha256"),
    }
    for field, expected_sha256 in source_text_contract.items():
        if _text_sha256(snapshot.get(field)) != expected_sha256:
            failures.append(f"SNAPSHOT_SOURCE_TEXT_{field.upper()}")
    parsed_snapshot_contract = {
        "retrieval_benchmark_source_text": "retrieval_benchmark",
        "development_scenarios_source_text": "development_scenarios",
        "qualification_holdout_source_text": "qualification_holdout",
    }
    for source_field, parsed_field in parsed_snapshot_contract.items():
        source_text = snapshot.get(source_field)
        if not isinstance(source_text, str):
            continue
        try:
            parsed = strict_json_loads(source_text)
        except Exception:
            failures.append(f"SNAPSHOT_SOURCE_JSON_{source_field.upper()}")
            continue
        if parsed != snapshot.get(parsed_field):
            failures.append(f"SNAPSHOT_PARSED_MISMATCH_{parsed_field.upper()}")
    if snapshot.get("holdout_labels_excluded_from_prompt") is not True:
        failures.append("HOLDOUT_LABEL_BOUNDARY")
    if snapshot.get("holdout_tuning_prohibited") is not True:
        failures.append("HOLDOUT_TUNING_BOUNDARY")
    holdout_snapshot = snapshot.get("qualification_holdout", {})
    if (
        holdout_snapshot.get("split") != "QUALIFICATION_HOLDOUT"
        or holdout_snapshot.get("tuning_prohibited") is not True
        or len(holdout_snapshot.get("scenarios", [])) != EXPECTED_UNIQUE_SCENARIOS
    ):
        failures.append("HOLDOUT_SNAPSHOT_CONTRACT")

    planner_metrics = pd.read_csv(root / "planner_mode_metrics.csv")
    if planner_metrics["mode"].tolist() != [PRODUCTION_CANDIDATE_MODE, DIAGNOSTIC_MODE]:
        failures.append("PLANNER_MODES")
    metric_rows = {str(row["mode"]): row for row in planner_metrics.to_dict(orient="records")}
    if planner_metrics.shape[0] != 2:
        failures.append("PLANNER_RUN_COUNTS")
    for mode in (PRODUCTION_CANDIDATE_MODE, DIAGNOSTIC_MODE):
        row = metric_rows.get(mode, {})
        if (
            row.get("unique_scenario_count") != EXPECTED_UNIQUE_SCENARIOS
            or row.get("seed_count") != len(EXPECTED_SEEDS_BY_MODE[mode])
            or row.get("run_count") != EXPECTED_RUNS_BY_MODE[mode]
        ):
            failures.append(f"PLANNER_RUN_COUNTS_{mode}")
    if int(planner_metrics["post_gate_unauthorized_action_count"].sum()) != 0:
        failures.append("POST_GATE_UNAUTHORIZED_ACTION")
    metric_columns = tuple(PLANNER_QUALITY_GATES) + (
        "safety_enforced_citation_precision",
        "safety_enforced_abstention_accuracy",
        "safety_enforced_action_accuracy",
        "hit_token_limit_rate",
    )
    for column in metric_columns:
        if not all(_finite_rate(value) for value in planner_metrics[column].tolist()):
            failures.append(f"PLANNER_METRIC_{column.upper()}")
    failures.extend(planner_quality_failures(planner_metrics))

    planner_results = require_json_object(
        strict_json_load(root / "planner_scenario_results.json"),
        source="stage-2 planner results",
    )
    records = planner_results.get("records", [])
    if (
        planner_results.get("unique_scenario_count") != EXPECTED_UNIQUE_SCENARIOS
        or planner_results.get("seed_count_by_mode")
        != {mode: len(seeds) for mode, seeds in EXPECTED_SEEDS_BY_MODE.items()}
        or planner_results.get("run_count_by_mode") != EXPECTED_RUNS_BY_MODE
        or len(records) != sum(EXPECTED_RUNS_BY_MODE.values())
    ):
        failures.append("PLANNER_RESULT_COUNT")
    identities = {(item.get("scenario_id"), item.get("mode"), item.get("seed")) for item in records}
    if len(identities) != len(records):
        failures.append("PLANNER_RESULT_IDENTITY")
    for mode, expected_seeds in EXPECTED_SEEDS_BY_MODE.items():
        mode_seeds = {item.get("seed") for item in records if item.get("mode") == mode}
        if mode_seeds != set(expected_seeds):
            failures.append(f"PLANNER_RESULT_SEEDS_{mode}")
    if any(item.get("post_gate_unauthorized") is not False for item in records):
        failures.append("PLANNER_RESULT_UNAUTHORIZED_ACTION")
    if any(
        "raw_output" in item or "thinking_content" in item or "reasoning_content" in item
        for item in records
    ):
        failures.append("CHAIN_OF_THOUGHT_PERSISTED")
    required_trace_fields = {
        "scenario_id",
        "query_id",
        "seed",
        "mode",
        "schema_valid",
        "protocol_error",
        "validation_errors",
        "validation_rule_ids",
        "invalid_final_response",
        "invalid_final_response_truncated",
        "parsed_candidate",
        "raw_action",
        "enforced_action",
        "enforced_evidence_ids",
        "expected_action",
        "expected_hold",
        "raw_action_correct",
        "raw_abstention_correct",
        "safety_action_correct",
        "safety_abstention_correct",
        "violations",
        "post_gate_unauthorized",
        "raw_citation_count",
        "raw_supported_citation_count",
        "filtered_citation_count",
        "safety_enforced_citation_count",
        "safety_enforced_supported_citation_count",
        "hit_token_limit",
        "finish_reason",
        "think_close_seen",
        "thinking_token_count",
        "final_token_count",
        "output_sha256",
        "final_response_char_count",
        "validated_packet",
    }
    if any(not required_trace_fields.issubset(item) for item in records):
        failures.append("PLANNER_TRACE_INCOMPLETE")
    if any(
        isinstance(item.get("invalid_final_response"), str)
        and (
            "<think>" in item["invalid_final_response"]
            or "</think>" in item["invalid_final_response"]
        )
        for item in records
    ):
        failures.append("CHAIN_OF_THOUGHT_IN_INVALID_FINAL_RESPONSE")
    if any(
        item.get("schema_valid") is True
        and (
            item.get("invalid_final_response") is not None
            or item.get("parsed_candidate") is not None
            or item.get("validation_rule_ids") != []
        )
        for item in records
    ):
        failures.append("VALID_PACKET_HAS_INVALID_DIAGNOSTICS")
    production_records = [item for item in records if item.get("mode") == PRODUCTION_CANDIDATE_MODE]
    if any(item.get("schema_valid") is not True for item in production_records):
        failures.append("PRODUCTION_PLANNER_SCHEMA_INVALID")
    if any(item.get("protocol_error") is not None for item in production_records):
        failures.append("PRODUCTION_PLANNER_PROTOCOL_ERROR")
    if any(item.get("validated_packet") is None for item in production_records):
        failures.append("PRODUCTION_PLANNER_PACKET_MISSING")

    for item in records:
        packet = item.get("validated_packet")
        raw_action_correct = isinstance(packet, dict) and packet.get("action_code") == item.get(
            "expected_action"
        )
        raw_abstention_correct = isinstance(packet, dict) and packet.get("abstain") == item.get(
            "expected_hold"
        )
        if item.get("raw_action_correct") is not raw_action_correct:
            failures.append("RAW_ACTION_CORRECTNESS_MISMATCH")
            break
        if item.get("raw_abstention_correct") is not raw_abstention_correct:
            failures.append("RAW_ABSTENTION_CORRECTNESS_MISMATCH")
            break
        if item.get("safety_action_correct") is not (
            item.get("enforced_action") == item.get("expected_action")
        ):
            failures.append("SAFETY_ACTION_CORRECTNESS_MISMATCH")
            break
        if item.get("safety_abstention_correct") is not (
            item.get("abstain") == item.get("expected_hold")
        ):
            failures.append("SAFETY_ABSTENTION_CORRECTNESS_MISMATCH")
            break
        enforced_ids = item.get("enforced_evidence_ids")
        if not isinstance(enforced_ids, list) or len(enforced_ids) != len(set(enforced_ids)):
            failures.append("ENFORCED_EVIDENCE_IDS_INVALID")
            break
        if item.get("enforced_action") == "HOLD_FOR_REVIEW":
            if enforced_ids:
                failures.append("HOLD_HAS_ENFORCED_EVIDENCE")
                break
        elif not enforced_ids:
            failures.append("ACTION_MISSING_ENFORCED_EVIDENCE")
            break
        raw_count = item.get("raw_citation_count")
        raw_supported_count = item.get("raw_supported_citation_count")
        enforced_count = item.get("safety_enforced_citation_count")
        enforced_supported_count = item.get("safety_enforced_supported_citation_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                raw_count,
                raw_supported_count,
                enforced_count,
                enforced_supported_count,
            )
        ):
            failures.append("CITATION_COUNTS_INVALID")
            break
        if raw_supported_count > raw_count or enforced_supported_count > enforced_count:
            failures.append("SUPPORTED_CITATION_COUNT_INVALID")
            break
        if enforced_count != len(enforced_ids):
            failures.append("ENFORCED_CITATION_COUNT_MISMATCH")
            break
        if isinstance(packet, dict):
            raw_ids = packet.get("evidence_ids")
            if (
                not isinstance(raw_ids, list)
                or len(raw_ids) != len(set(raw_ids))
                or raw_count != len(raw_ids)
            ):
                failures.append("RAW_CITATION_COUNT_MISMATCH")
                break
        elif raw_count != 0 or raw_supported_count != 0:
            failures.append("INVALID_PACKET_HAS_RAW_CITATIONS")
            break
        validation_errors = item.get("validation_errors")
        validation_rule_ids = item.get("validation_rule_ids")
        if not isinstance(validation_errors, list) or not isinstance(validation_rule_ids, list):
            failures.append("VALIDATION_DIAGNOSTICS_INVALID")
            break
        error_rule_ids: set[str] = set()
        for error in validation_errors:
            if (
                not isinstance(error, dict)
                or not {"field", "type", "message", "rule_id", "context"}.issubset(error)
                or not isinstance(error.get("rule_id"), str)
            ):
                failures.append("VALIDATION_ERROR_DETAIL_INVALID")
                break
            error_rule_ids.add(error["rule_id"])
        else:
            if error_rule_ids and sorted(error_rule_ids) != validation_rule_ids:
                failures.append("VALIDATION_RULE_IDS_MISMATCH")
                break
            if item.get("protocol_error") == "SCHEMA_VALIDATION_ERROR" and not error_rule_ids:
                failures.append("SCHEMA_VALIDATION_DIAGNOSTICS_MISSING")
                break
            final_chars = item.get("final_response_char_count")
            invalid_final = item.get("invalid_final_response")
            if packet is None and isinstance(final_chars, int) and final_chars > 0:
                if not isinstance(invalid_final, str) or not invalid_final:
                    failures.append("INVALID_FINAL_RESPONSE_MISSING")
                    break
            elif invalid_final is not None:
                failures.append("INVALID_FINAL_RESPONSE_UNEXPECTED")
                break

    for mode in (PRODUCTION_CANDIDATE_MODE, DIAGNOSTIC_MODE):
        recomputed = _recomputed_mode_metrics(records, mode)
        row = metric_rows.get(mode, {})
        for metric, expected in recomputed.items():
            actual = row.get(metric)
            if metric == "post_gate_unauthorized_action_count":
                if int(actual) != int(expected):
                    failures.append(f"PLANNER_RECOMPUTE_{mode}_{metric.upper()}")
            elif not _close(actual, expected):
                failures.append(f"PLANNER_RECOMPUTE_{mode}_{metric.upper()}")

    summary = require_json_object(
        strict_json_load(root / "stage2_summary.json"), source="stage-2 summary"
    )
    expected_summary_failures = planner_quality_failures(planner_metrics)
    if int(planner_metrics["post_gate_unauthorized_action_count"].sum()) != 0:
        expected_summary_failures.append("POST_GATE_UNAUTHORIZED_ACTION")
    if summary.get("planner_quality_gate_failures") != expected_summary_failures:
        failures.append("SUMMARY_QUALITY_FAILURES")
    expected_summary_status = "PASS" if not expected_summary_failures else "FAIL"
    if summary.get("status") != expected_summary_status:
        failures.append("SUMMARY_QUALITY_STATUS")
    if summary.get("status") != "PASS":
        failures.append("SUMMARY_STATUS")
    if summary.get("production_claim_permitted") is not False:
        failures.append("PRODUCTION_CLAIM_BOUNDARY")
    if summary.get("production_candidate_mode") != PRODUCTION_CANDIDATE_MODE:
        failures.append("PRODUCTION_CANDIDATE_MODE")
    if summary.get("thinking_mode_status") != "DIAGNOSTIC_ONLY":
        failures.append("THINKING_MODE_STATUS")
    if summary.get("tuning_on_qualification_holdout") is not False:
        failures.append("HOLDOUT_TUNING_BOUNDARY")

    return {
        "schema_version": "4.0",
        "artifact_dir": str(root),
        "artifact_manifest_sha256": sha256_file(root / "artifact_manifest.json"),
        "checked_files": integrity["checked"],
        "retrieval_query_count": query_results.get("query_count"),
        "planner_unique_scenario_count": planner_results.get("unique_scenario_count"),
        "planner_seed_count_by_mode": planner_results.get("seed_count_by_mode"),
        "planner_run_count_by_mode": planner_results.get("run_count_by_mode"),
        "post_gate_unauthorized_action_count": int(
            planner_metrics["post_gate_unauthorized_action_count"].sum()
        ),
        "production_candidate_mode": PRODUCTION_CANDIDATE_MODE,
        "diagnostic_mode": DIAGNOSTIC_MODE,
        "planner_quality_gates": PLANNER_QUALITY_GATES,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
