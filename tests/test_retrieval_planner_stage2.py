from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from finops_r.artifacts import artifact_manifest, atomic_csv, atomic_json
from finops_r.planner_stage2 import (
    DEFAULT_DIAGNOSTIC_SEEDS,
    DEFAULT_EVALUATION_SEEDS,
    DEVELOPMENT_SCENARIOS_SHA256,
    NON_THINKING_MAX_NEW_TOKENS,
    PlannerScenarioBundle,
    QUALIFICATION_HOLDOUT_SHA256,
    THINKING_MAX_NEW_TOKENS,
    build_stage2_qwen_messages,
    exact_snapshot_object,
    extract_final_response,
    load_planner_scenarios,
    profile_planner,
    split_qwen_generated_tokens,
)
from finops_r.policy_cards import compile_policy_cards, load_compiled_policy_cards
from finops_r.retrieval import tokenize
from finops_r.retrieval_stage2 import (
    BGE_MODEL_REVISION,
    QWEN_MODEL_REVISION,
    STAGE2_BENCHMARK_SHA256,
    Stage2Benchmark,
    build_stage2_documents,
    compare_retrievers,
    load_stage2_benchmark,
    ranking_metrics,
)
from finops_r.stage2_verification import (
    PLANNER_QUALITY_GATES,
    STAGE2_IMPLEMENTATION_REVISION,
    verify_stage2_artifact,
)


ROOT = Path(__file__).parents[1]
SOURCE_CARDS = ROOT / "data" / "policy_cards" / "source_cards.json"
BENCHMARK = ROOT / "data" / "policy_cards" / "retrieval_benchmark_stage2.json"
SCENARIOS = ROOT / "data" / "policy_cards" / "planner_scenarios_stage2.json"
HOLDOUT = ROOT / "data" / "policy_cards" / "planner_scenarios_stage2_v5_holdout.json"


class HashEncoder:
    dimension = 64

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=float)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:2], "big") % self.dimension
                vectors[row, index] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm > 0.0:
                vectors[row] /= norm
        return vectors

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


class InvalidEncoder(HashEncoder):
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 2), dtype=float)


class FakeGenerator:
    def generate(
        self, messages: list[dict[str, str]], *, mode: str, seed: int
    ) -> dict[str, object]:
        del seed
        payload = json.loads(messages[1]["content"])
        request = payload["request"]
        evidence = payload["retrieved_evidence"]
        candidate = request["candidate_action"]
        permitted = request["permitted_actions"]
        supporting = [
            item["document_id"]
            for item in evidence
            if item["source_type"] == "POLICY_CARD" and candidate in item["allowed_actions"]
        ]
        if candidate in permitted and supporting:
            action = candidate
            evidence_ids = supporting[:1]
            abstain = False
            reason = "SUPPORTED_ACTION"
            confidence = 1.0
        else:
            action = "HOLD_FOR_REVIEW"
            evidence_ids = []
            abstain = True
            reason = (
                "ACTION_NOT_PERMITTED" if candidate not in permitted else "INSUFFICIENT_EVIDENCE"
            )
            confidence = 0.0
        packet = {
            "schema_version": "1.0",
            "case_id": request["case_id"],
            "decision_time": request["decision_time"],
            "root_cause": request["predicted_cause"],
            "action_code": action,
            "evidence_ids": evidence_ids,
            "confidence": confidence,
            "abstain": abstain,
            "reason_code": reason,
            "model_id": "fake-stage2",
            "prompt_version": "finops_action_packet_v1",
        }
        final = json.dumps(packet, separators=(",", ":"), sort_keys=True)
        text = f"<think>not persisted</think>\n{final}" if mode == "THINKING" else final
        return {
            "text": text,
            "latency_ms": 1.0,
            "input_tokens": 100,
            "output_tokens": 50,
            "peak_vram_bytes": 1024,
            "hit_token_limit": False,
            "finish_reason": "eos",
            "think_close_seen": mode == "THINKING",
            "thinking_token_count": 10 if mode == "THINKING" else 0,
            "final_token_count": 40 if mode == "THINKING" else 50,
        }


class InvalidSchemaGenerator:
    def generate(
        self, messages: list[dict[str, str]], *, mode: str, seed: int
    ) -> dict[str, object]:
        del messages, seed
        final = '{"schema_version":"wrong"}'
        text = f"<think>not persisted</think>\n{final}" if mode == "THINKING" else final
        return {
            "text": text,
            "latency_ms": 1.0,
            "input_tokens": 100,
            "output_tokens": 5,
            "peak_vram_bytes": 1024,
            "hit_token_limit": False,
            "finish_reason": "eos",
            "think_close_seen": mode == "THINKING",
            "thinking_token_count": 2 if mode == "THINKING" else 0,
            "final_token_count": 3 if mode == "THINKING" else 5,
        }


class OverCitingGenerator(FakeGenerator):
    def generate(
        self, messages: list[dict[str, str]], *, mode: str, seed: int
    ) -> dict[str, object]:
        generated = super().generate(messages, mode=mode, seed=seed)
        text = str(generated["text"])
        final = text.split("</think>", maxsplit=1)[-1].strip()
        packet = json.loads(final)
        if packet["action_code"] != "HOLD_FOR_REVIEW":
            prompt = json.loads(messages[1]["content"])
            packet["evidence_ids"] = [item["document_id"] for item in prompt["retrieved_evidence"]]
        final = json.dumps(packet, separators=(",", ":"), sort_keys=True)
        generated["text"] = (
            f"<think>not persisted</think>\n{final}" if mode == "THINKING" else final
        )
        return generated


class RetrievalPlannerStage2Tests(unittest.TestCase):
    def _fixture(self) -> tuple[Stage2Benchmark, list[object]]:
        benchmark = load_stage2_benchmark(BENCHMARK)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compile_policy_cards(SOURCE_CARDS, root)
            cards = load_compiled_policy_cards(root / "policy_cards.json")
            documents = build_stage2_documents(cards, benchmark)
        return benchmark, documents

    def test_locked_files_have_expected_sha256(self) -> None:
        self.assertEqual(
            hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(), STAGE2_BENCHMARK_SHA256
        )
        self.assertEqual(
            hashlib.sha256(SCENARIOS.read_bytes()).hexdigest(),
            DEVELOPMENT_SCENARIOS_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(HOLDOUT.read_bytes()).hexdigest(),
            QUALIFICATION_HOLDOUT_SHA256,
        )

    def test_locked_benchmark_has_complete_categories_and_counts(self) -> None:
        benchmark = load_stage2_benchmark(BENCHMARK)
        self.assertEqual(len(benchmark.queries), 35)
        self.assertEqual(len(benchmark.distractors), 14)
        self.assertTrue(benchmark.tuning_prohibited)
        self.assertEqual(tuple(benchmark.top_k_values), (1, 3, 5))

    def test_benchmark_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            path.write_bytes(BENCHMARK.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_stage2_benchmark(path)

    def test_relevant_documents_exist_and_are_eligible(self) -> None:
        benchmark, documents = self._fixture()
        self.assertEqual(len(documents), 21)
        by_id = {item.document_id: item for item in documents}
        for query in benchmark.queries:
            for document_id in query.relevant_ids:
                self.assertIn(document_id, by_id)
                self.assertTrue(by_id[document_id].eligible_at(query.decision_time))

    def test_future_distractor_is_ineligible(self) -> None:
        benchmark, documents = self._fixture()
        query = next(
            item for item in benchmark.queries if item.query_id == "time_future_distractor"
        )
        future = next(
            item for item in documents if item.document_id == "benchmark.case.future_tplus0"
        )
        self.assertFalse(future.eligible_at(query.decision_time))

    def test_ranking_metrics_match_hand_calculation(self) -> None:
        result = ranking_metrics(["x", "a", "b"], ("a", "b"))
        self.assertEqual(result["recall_at_1"], 0.0)
        self.assertEqual(result["recall_at_3"], 1.0)
        self.assertEqual(result["mrr"], 0.5)
        self.assertGreater(result["ndcg_at_3"], 0.0)
        self.assertLess(result["ndcg_at_3"], 1.0)

    def test_ranking_metrics_reject_duplicate_hits(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ranking_metrics(["a", "a"], ("a",))

    def test_three_retrievers_cover_all_queries_without_time_leakage(self) -> None:
        benchmark, documents = self._fixture()
        candidates, records = compare_retrievers(documents, benchmark, HashEncoder())
        self.assertEqual(
            [item["method"] for item in candidates], ["BM25", "BGE_DENSE", "RRF_HYBRID"]
        )
        self.assertEqual(len(records), 105)
        self.assertEqual(sum(item["future_evidence_count"] for item in records), 0)
        self.assertTrue(all(item["query_count"] == 35 for item in candidates))

    def test_dense_encoder_shape_is_enforced(self) -> None:
        benchmark, documents = self._fixture()
        with self.assertRaisesRegex(ValueError, "invalid matrix"):
            compare_retrievers(documents, benchmark, InvalidEncoder())

    def test_planner_scenarios_are_locked_and_consistent(self) -> None:
        bundle = load_planner_scenarios(SCENARIOS)
        self.assertEqual(len(bundle.scenarios), 13)
        self.assertEqual(sum(item.expected_hold for item in bundle.scenarios), 7)
        self.assertEqual(bundle.split, "DEVELOPMENT_REGRESSION")
        self.assertFalse(bundle.tuning_prohibited)

        holdout = load_planner_scenarios(HOLDOUT)
        self.assertEqual(len(holdout.scenarios), 12)
        self.assertEqual(sum(item.expected_hold for item in holdout.scenarios), 6)
        self.assertEqual(holdout.split, "QUALIFICATION_HOLDOUT")
        self.assertTrue(holdout.tuning_prohibited)

    def test_planner_scenario_rejects_inconsistent_hold(self) -> None:
        payload = json.loads(SCENARIOS.read_text(encoding="utf-8"))
        payload["scenarios"][0]["expected_hold"] = True
        with self.assertRaises(ValidationError):
            PlannerScenarioBundle.model_validate(payload)

    def test_exact_snapshot_preserves_absent_optional_fields(self) -> None:
        source_text = SCENARIOS.read_text(encoding="utf-8")
        snapshot = exact_snapshot_object(source_text)
        self.assertEqual(snapshot, json.loads(source_text))
        self.assertNotIn("locked_at_utc", snapshot)

    def test_thinking_protocol_extracts_only_final_response(self) -> None:
        final, error = extract_final_response(
            '<think>private reasoning</think>\n{"ok":1}', "THINKING"
        )
        self.assertEqual(final, '{"ok":1}')
        self.assertIsNone(error)
        final, error = extract_final_response("<think>unfinished", "THINKING")
        self.assertIsNone(final)
        self.assertEqual(error, "UNCLOSED_THINK_BLOCK")

    def test_non_thinking_protocol_rejects_think_blocks(self) -> None:
        final, error = extract_final_response("<think>x</think>{}", "NON_THINKING")
        self.assertIsNone(final)
        self.assertEqual(error, "UNEXPECTED_THINK_BLOCK")

    def test_qwen_thinking_tokens_are_split_before_special_token_decode(self) -> None:
        final_ids, error = split_qwen_generated_tokens(
            [151667, 101, 102, 151668, 201, 202],
            "THINKING",
            think_open_token_id=151667,
            think_close_token_id=151668,
        )
        self.assertEqual(final_ids, [201, 202])
        self.assertIsNone(error)

        final_ids, error = split_qwen_generated_tokens(
            [151667, 101, 102],
            "THINKING",
            think_open_token_id=151667,
            think_close_token_id=151668,
        )
        self.assertIsNone(final_ids)
        self.assertEqual(error, "UNCLOSED_THINK_BLOCK")

        final_ids, error = split_qwen_generated_tokens(
            [201, 151668, 202],
            "NON_THINKING",
            think_open_token_id=151667,
            think_close_token_id=151668,
        )
        self.assertIsNone(final_ids)
        self.assertEqual(error, "UNEXPECTED_THINK_BLOCK")

    def test_fake_planner_exercises_both_modes_and_fail_closed_gate(self) -> None:
        benchmark, documents = self._fixture()
        _, retrieval_records = compare_retrievers(documents, benchmark, HashEncoder())
        metrics, records = profile_planner(
            load_planner_scenarios(SCENARIOS),
            retrieval_records,
            documents,
            FakeGenerator(),
        )
        self.assertEqual([item["mode"] for item in metrics], ["NON_THINKING", "THINKING"])
        self.assertEqual(len(records), 26)
        self.assertTrue(all(item["schema_valid"] for item in records))
        self.assertEqual(sum(item["post_gate_unauthorized"] for item in records), 0)
        self.assertTrue(all("raw_output" not in item for item in records))
        self.assertTrue(all(item["raw_model_action_accuracy"] == 12 / 13 for item in metrics))
        self.assertTrue(all(item["safety_enforced_action_accuracy"] == 12 / 13 for item in metrics))

    def test_multi_seed_profile_has_complete_identity_and_counts(self) -> None:
        benchmark, documents = self._fixture()
        _, retrieval_records = compare_retrievers(documents, benchmark, HashEncoder())
        metrics, records = profile_planner(
            load_planner_scenarios(HOLDOUT),
            retrieval_records,
            documents,
            FakeGenerator(),
            production_seeds=DEFAULT_EVALUATION_SEEDS,
            diagnostic_seeds=DEFAULT_DIAGNOSTIC_SEEDS,
        )
        self.assertEqual(len(records), 48)
        self.assertEqual(
            len({(item["scenario_id"], item["mode"], item["seed"]) for item in records}),
            48,
        )
        by_mode = {item["mode"]: item for item in metrics}
        self.assertEqual(by_mode["NON_THINKING"]["seed_count"], 3)
        self.assertEqual(by_mode["NON_THINKING"]["run_count"], 36)
        self.assertEqual(by_mode["THINKING"]["seed_count"], 1)
        self.assertEqual(by_mode["THINKING"]["run_count"], 12)

    def test_safety_enforcement_removes_non_supporting_citations(self) -> None:
        benchmark, documents = self._fixture()
        _, retrieval_records = compare_retrievers(documents, benchmark, HashEncoder())
        metrics, records = profile_planner(
            load_planner_scenarios(SCENARIOS),
            retrieval_records,
            documents,
            OverCitingGenerator(),
        )
        supported = [item for item in records if item["scenario_id"] == "supported_allocation"]
        self.assertTrue(supported)
        self.assertTrue(
            all(item["raw_citation_count"] > item["filtered_citation_count"] for item in supported)
        )
        self.assertTrue(
            all(
                len(item["enforced_evidence_ids"]) == item["filtered_citation_count"]
                for item in supported
            )
        )
        self.assertTrue(all(item["raw_model_citation_precision"] < 1.0 for item in metrics))
        self.assertTrue(all(item["safety_enforced_citation_precision"] == 1.0 for item in metrics))

    def test_stage2_prompt_exposes_complete_contract(self) -> None:
        scenario = load_planner_scenarios(SCENARIOS).scenarios[0]
        messages = build_stage2_qwen_messages(scenario.request(), [])
        payload = json.loads(messages[1]["content"])
        constraints = payload["output_contract"]["field_constraints"]
        self.assertEqual(constraints["schema_version"]["const"], "1.0")
        self.assertEqual(constraints["case_id"]["const"], scenario.case_id)
        self.assertEqual(constraints["root_cause"]["const"], scenario.predicted_cause)
        self.assertEqual(constraints["prompt_version"]["const"], "finops_action_packet_v1")
        self.assertEqual(constraints["action_code"]["enum"], ["HOLD_FOR_REVIEW"])
        self.assertEqual(constraints["evidence_ids"], {"const": []})

    def test_prompt_precheck_allows_only_supporting_policy_citations(self) -> None:
        benchmark, documents = self._fixture()
        _, retrieval_records = compare_retrievers(documents, benchmark, HashEncoder())
        scenario = next(
            item
            for item in load_planner_scenarios(HOLDOUT).scenarios
            if item.scenario_id == "v5q_supported_affirmation"
        )
        rrf = next(
            item
            for item in retrieval_records
            if item["method"] == "RRF_HYBRID" and item["query_id"] == scenario.query_id
        )
        by_id = {item.document_id: item for item in documents}
        evidence = [
            (
                type(
                    "Hit",
                    (),
                    {
                        "document_id": document_id,
                        "source_type": by_id[document_id].source_type,
                        "allowed_actions": by_id[document_id].allowed_actions,
                    },
                )(),
                by_id[document_id].text,
            )
            for document_id in rrf["hit_ids"]
        ]
        payload = json.loads(build_stage2_qwen_messages(scenario.request(), evidence)[1]["content"])
        precheck = payload["output_contract"]["deterministic_policy_precheck"]
        self.assertTrue(precheck["candidate_is_eligible"])
        self.assertTrue(precheck["supporting_policy_card_ids"])
        self.assertEqual(
            payload["output_contract"]["field_constraints"]["evidence_ids"]["allowed_values"],
            precheck["supporting_policy_card_ids"],
        )

    def test_unpermitted_candidate_forces_empty_citation_contract(self) -> None:
        benchmark, documents = self._fixture()
        _, retrieval_records = compare_retrievers(documents, benchmark, HashEncoder())
        scenario = next(
            item
            for item in load_planner_scenarios(SCENARIOS).scenarios
            if item.scenario_id == "regression_v4_unpermitted_affirmation"
        )
        rrf = next(
            item
            for item in retrieval_records
            if item["method"] == "RRF_HYBRID" and item["query_id"] == scenario.query_id
        )
        by_id = {item.document_id: item for item in documents}
        evidence = [
            (
                type(
                    "Hit",
                    (),
                    {
                        "document_id": document_id,
                        "source_type": by_id[document_id].source_type,
                        "allowed_actions": by_id[document_id].allowed_actions,
                    },
                )(),
                by_id[document_id].text,
            )
            for document_id in rrf["hit_ids"]
        ]
        payload = json.loads(build_stage2_qwen_messages(scenario.request(), evidence)[1]["content"])
        contract = payload["output_contract"]
        precheck = contract["deterministic_policy_precheck"]
        self.assertFalse(precheck["candidate_is_permitted"])
        self.assertFalse(precheck["candidate_is_eligible"])
        self.assertTrue(precheck["supporting_policy_card_ids"])
        self.assertEqual(precheck["required_action"], "HOLD_FOR_REVIEW")
        self.assertEqual(precheck["required_reason_code"], "ACTION_NOT_PERMITTED")
        self.assertEqual(contract["field_constraints"]["evidence_ids"], {"const": []})

    def test_invalid_output_records_safe_diagnostics(self) -> None:
        benchmark, documents = self._fixture()
        _, retrieval_records = compare_retrievers(documents, benchmark, HashEncoder())
        metrics, records = profile_planner(
            load_planner_scenarios(SCENARIOS),
            retrieval_records,
            documents,
            InvalidSchemaGenerator(),
        )
        self.assertTrue(
            all(item["protocol_error"] == "SCHEMA_VALIDATION_ERROR" for item in records)
        )
        self.assertTrue(all(item["validation_errors"] for item in records))
        self.assertTrue(all(item["validation_rule_ids"] for item in records))
        self.assertTrue(all(item["invalid_final_response"] for item in records))
        self.assertTrue(all(item["parsed_candidate"] for item in records))
        self.assertTrue(all("raw_output" not in item for item in records))
        self.assertTrue(all(item["validated_packet"] is None for item in records))
        self.assertTrue(all(item["raw_model_action_accuracy"] == 0.0 for item in metrics))
        self.assertTrue(all(item["raw_model_abstention_accuracy"] == 0.0 for item in metrics))
        self.assertTrue(all(item["safety_enforced_action_accuracy"] == 7 / 13 for item in metrics))

    def test_model_revisions_are_full_commit_shas(self) -> None:
        for revision in (BGE_MODEL_REVISION, QWEN_MODEL_REVISION):
            self.assertEqual(len(revision), 40)
            int(revision, 16)

    def _write_valid_artifact(self, root: Path) -> None:
        policy = compile_policy_cards(SOURCE_CARDS, root)
        candidates = []
        for method in ("BM25", "BGE_DENSE", "RRF_HYBRID"):
            candidates.append(
                {
                    "method": method,
                    "query_count": 35,
                    "future_evidence_count": 0,
                    "recall_at_1": 0.5,
                    "recall_at_3": 0.8,
                    "recall_at_5": 1.0,
                    "mrr": 0.7,
                    "ndcg_at_5": 0.8,
                }
            )
        atomic_csv(root / "retrieval_candidate_comparison.csv", pd.DataFrame(candidates))
        atomic_json(
            root / "retrieval_query_results.json",
            {
                "query_count": 35,
                "future_evidence_count": 0,
                "records": [{"record": index} for index in range(105)],
            },
        )
        planner_rows = []
        for mode, seeds in {
            "NON_THINKING": DEFAULT_EVALUATION_SEEDS,
            "THINKING": DEFAULT_DIAGNOSTIC_SEEDS,
        }.items():
            planner_rows.append(
                {
                    "mode": mode,
                    "unique_scenario_count": 12,
                    "seed_count": len(seeds),
                    "run_count": 12 * len(seeds),
                    "post_gate_unauthorized_action_count": 0,
                    "schema_valid_rate": 1.0,
                    "raw_model_citation_precision": 1.0,
                    "safety_enforced_citation_precision": 1.0,
                    "raw_model_abstention_accuracy": 1.0,
                    "raw_model_action_accuracy": 1.0,
                    "per_seed_min_raw_model_abstention_accuracy": 1.0,
                    "per_seed_min_raw_model_action_accuracy": 1.0,
                    "safety_enforced_abstention_accuracy": 1.0,
                    "safety_enforced_action_accuracy": 1.0,
                    "hit_token_limit_rate": 0.0,
                }
            )
        atomic_csv(root / "planner_mode_metrics.csv", pd.DataFrame(planner_rows))
        holdout = load_planner_scenarios(HOLDOUT)
        records = []
        for mode, seeds in {
            "NON_THINKING": DEFAULT_EVALUATION_SEEDS,
            "THINKING": DEFAULT_DIAGNOSTIC_SEEDS,
        }.items():
            for seed in seeds:
                for scenario in holdout.scenarios:
                    citation_count = 0 if scenario.expected_hold else 1
                    packet = {
                        "schema_version": "1.0",
                        "action_code": scenario.expected_action,
                        "abstain": scenario.expected_hold,
                        "evidence_ids": (
                            [] if scenario.expected_hold else ["synthetic.policy.card"]
                        ),
                    }
                    records.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "query_id": scenario.query_id,
                            "seed": seed,
                            "mode": mode,
                            "schema_valid": True,
                            "protocol_error": None,
                            "validation_errors": [],
                            "validation_rule_ids": [],
                            "invalid_final_response": None,
                            "invalid_final_response_truncated": False,
                            "parsed_candidate": None,
                            "raw_action": scenario.expected_action,
                            "enforced_action": scenario.expected_action,
                            "enforced_evidence_ids": (
                                [] if scenario.expected_hold else ["synthetic.policy.card"]
                            ),
                            "expected_action": scenario.expected_action,
                            "expected_hold": scenario.expected_hold,
                            "raw_action_correct": True,
                            "raw_abstention_correct": True,
                            "safety_action_correct": True,
                            "safety_abstention_correct": True,
                            "abstain": scenario.expected_hold,
                            "violations": [],
                            "raw_unauthorized": False,
                            "raw_unsupported": False,
                            "post_gate_unauthorized": False,
                            "raw_citation_count": citation_count,
                            "raw_supported_citation_count": citation_count,
                            "filtered_citation_count": citation_count,
                            "safety_enforced_citation_count": citation_count,
                            "safety_enforced_supported_citation_count": citation_count,
                            "hit_token_limit": False,
                            "finish_reason": "eos",
                            "think_close_seen": mode == "THINKING",
                            "thinking_token_count": 100 if mode == "THINKING" else 0,
                            "final_token_count": 50,
                            "output_sha256": "0" * 64,
                            "final_response_char_count": 100,
                            "validated_packet": packet,
                        }
                    )
        atomic_json(
            root / "planner_scenario_results.json",
            {
                "unique_scenario_count": 12,
                "seed_count_by_mode": {
                    "NON_THINKING": len(DEFAULT_EVALUATION_SEEDS),
                    "THINKING": len(DEFAULT_DIAGNOSTIC_SEEDS),
                },
                "run_count_by_mode": {
                    "NON_THINKING": 12 * len(DEFAULT_EVALUATION_SEEDS),
                    "THINKING": 12 * len(DEFAULT_DIAGNOSTIC_SEEDS),
                },
                "records": records,
            },
        )
        atomic_json(
            root / "evaluation_input_snapshot.json",
            {
                "retrieval_benchmark_sha256": STAGE2_BENCHMARK_SHA256,
                "development_scenarios_sha256": DEVELOPMENT_SCENARIOS_SHA256,
                "qualification_holdout_sha256": QUALIFICATION_HOLDOUT_SHA256,
                "source_cards_sha256": hashlib.sha256(SOURCE_CARDS.read_bytes()).hexdigest(),
                "retrieval_benchmark_source_text": BENCHMARK.read_text(encoding="utf-8"),
                "development_scenarios_source_text": SCENARIOS.read_text(encoding="utf-8"),
                "qualification_holdout_source_text": HOLDOUT.read_text(encoding="utf-8"),
                "source_cards_source_text": SOURCE_CARDS.read_text(encoding="utf-8"),
                "retrieval_benchmark": json.loads(BENCHMARK.read_text(encoding="utf-8")),
                "development_scenarios": json.loads(SCENARIOS.read_text(encoding="utf-8")),
                "qualification_holdout": json.loads(HOLDOUT.read_text(encoding="utf-8")),
                "holdout_labels_excluded_from_prompt": True,
                "holdout_tuning_prohibited": True,
            },
        )
        atomic_json(
            root / "runtime_manifest.json",
            {
                "implementation_revision": STAGE2_IMPLEMENTATION_REVISION,
                "benchmark_sha256": STAGE2_BENCHMARK_SHA256,
                "development_scenarios_sha256": DEVELOPMENT_SCENARIOS_SHA256,
                "qualification_holdout_sha256": QUALIFICATION_HOLDOUT_SHA256,
                "source_cards_sha256": hashlib.sha256(SOURCE_CARDS.read_bytes()).hexdigest(),
                "models": {
                    "bge_revision": BGE_MODEL_REVISION,
                    "qwen_revision": QWEN_MODEL_REVISION,
                },
                "generation": {
                    "production_evaluation_seeds": list(DEFAULT_EVALUATION_SEEDS),
                    "diagnostic_evaluation_seeds": list(DEFAULT_DIAGNOSTIC_SEEDS),
                    "thinking": {
                        "max_new_tokens": THINKING_MAX_NEW_TOKENS,
                        "role": "DIAGNOSTIC_ONLY",
                    },
                    "non_thinking": {
                        "max_new_tokens": NON_THINKING_MAX_NEW_TOKENS,
                        "role": "PRODUCTION_CANDIDATE",
                    },
                },
                "external_execution": False,
                "chain_of_thought_persisted": False,
                "compiled_policy_cards_sha256": policy["compiled_sha256"],
            },
        )
        atomic_json(
            root / "stage2_summary.json",
            {
                "status": "PASS",
                "planner_quality_gates": PLANNER_QUALITY_GATES,
                "planner_quality_gate_failures": [],
                "production_claim_permitted": False,
                "production_candidate_mode": "NON_THINKING",
                "thinking_mode_status": "DIAGNOSTIC_ONLY",
                "tuning_on_qualification_holdout": False,
            },
        )
        atomic_json(root / "artifact_manifest.json", artifact_manifest(root))

    def test_independent_verifier_accepts_complete_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_valid_artifact(root)
            result = verify_stage2_artifact(root)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["checked_files"], 9)

    def test_independent_verifier_keeps_thinking_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_valid_artifact(root)
            payload = json.loads(
                (root / "planner_scenario_results.json").read_text(encoding="utf-8")
            )
            for item in payload["records"]:
                if item["mode"] != "THINKING":
                    continue
                item["schema_valid"] = False
                item["protocol_error"] = "UNCLOSED_THINK_BLOCK"
                item["raw_action"] = None
                item["enforced_action"] = "HOLD_FOR_REVIEW"
                item["enforced_evidence_ids"] = []
                item["raw_action_correct"] = False
                item["raw_abstention_correct"] = False
                item["safety_action_correct"] = item["expected_action"] == "HOLD_FOR_REVIEW"
                item["safety_abstention_correct"] = item["expected_hold"]
                item["abstain"] = True
                item["violations"] = ["INVALID_MODEL_OUTPUT"]
                item["raw_citation_count"] = 0
                item["raw_supported_citation_count"] = 0
                item["filtered_citation_count"] = 0
                item["safety_enforced_citation_count"] = 0
                item["safety_enforced_supported_citation_count"] = 0
                item["hit_token_limit"] = True
                item["finish_reason"] = "length"
                item["think_close_seen"] = False
                item["thinking_token_count"] = THINKING_MAX_NEW_TOKENS
                item["final_token_count"] = 0
                item["final_response_char_count"] = 0
                item["validated_packet"] = None
            atomic_json(root / "planner_scenario_results.json", payload)
            metrics = pd.read_csv(root / "planner_mode_metrics.csv")
            row = metrics["mode"] == "THINKING"
            metrics.loc[row, "schema_valid_rate"] = 0.0
            metrics.loc[row, "raw_model_citation_precision"] = 1.0
            metrics.loc[row, "safety_enforced_citation_precision"] = 1.0
            metrics.loc[row, "raw_model_abstention_accuracy"] = 0.0
            metrics.loc[row, "raw_model_action_accuracy"] = 0.0
            metrics.loc[row, "per_seed_min_raw_model_abstention_accuracy"] = 0.0
            metrics.loc[row, "per_seed_min_raw_model_action_accuracy"] = 0.0
            metrics.loc[row, "safety_enforced_abstention_accuracy"] = 0.5
            metrics.loc[row, "safety_enforced_action_accuracy"] = 0.5
            metrics.loc[row, "hit_token_limit_rate"] = 1.0
            atomic_csv(root / "planner_mode_metrics.csv", metrics)
            atomic_json(root / "artifact_manifest.json", artifact_manifest(root))
            result = verify_stage2_artifact(root)
            self.assertEqual(result["status"], "PASS")

    def test_independent_verifier_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_valid_artifact(root)
            (root / "stage2_summary.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "(size|checksum) mismatch"):
                verify_stage2_artifact(root)

    def test_independent_verifier_rejects_pre_v2_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_valid_artifact(root)
            runtime = json.loads((root / "runtime_manifest.json").read_text(encoding="utf-8"))
            runtime.pop("implementation_revision")
            atomic_json(root / "runtime_manifest.json", runtime)
            atomic_json(root / "artifact_manifest.json", artifact_manifest(root))
            result = verify_stage2_artifact(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("IMPLEMENTATION_REVISION", result["failures"])

    def test_independent_verifier_rejects_planner_quality_below_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_valid_artifact(root)
            metrics = pd.read_csv(root / "planner_mode_metrics.csv")
            metrics.loc[0, "schema_valid_rate"] = 0.99
            metrics.loc[0, "raw_model_action_accuracy"] = 0.5
            metrics.loc[1, "raw_model_action_accuracy"] = 0.0
            atomic_csv(root / "planner_mode_metrics.csv", metrics)
            atomic_json(root / "artifact_manifest.json", artifact_manifest(root))
            result = verify_stage2_artifact(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("PLANNER_QUALITY_NON_THINKING_SCHEMA_VALID_RATE", result["failures"])
            self.assertIn(
                "PLANNER_QUALITY_NON_THINKING_RAW_MODEL_ACTION_ACCURACY",
                result["failures"],
            )
            self.assertNotIn(
                "PLANNER_QUALITY_THINKING_RAW_MODEL_ACTION_ACCURACY",
                result["failures"],
            )


if __name__ == "__main__":
    unittest.main()
