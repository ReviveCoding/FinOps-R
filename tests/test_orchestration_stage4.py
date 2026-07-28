from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from finops_r.checkpointing_stage4 import (
    CheckpointConflictError,
    ReleasedTraceError,
    SQLiteCheckpointStore,
)
from finops_r.langgraph_stage4 import LangGraphUnavailableError, build_langgraph_stage4
from finops_r.orchestration_stage4 import (
    DurableStage4Orchestrator,
    InjectedCrash,
    Stage4Inputs,
    default_stage4_inputs,
    run_stage4_reference,
)
from finops_r.planner import ActionPacket, PlannerRequest, hold_packet
from finops_r.policy_cards import compile_policy_cards, load_compiled_policy_cards
from finops_r.retrieval import RetrievalHit
from finops_r.stage4_qualification import (
    run_stage4_qualification,
    verify_stage4_qualification,
)
from finops_r.verification_policy import CLAIM_BOUNDARY, PolicyContext
from finops_r.verifier_stage4 import (
    load_frozen_verifier,
    verify_frozen_verifier_provenance,
)
from scripts.freeze_learned_verifier_stage4 import freeze

ROOT = Path(__file__).parents[1]
SOURCE_CARDS = ROOT / "data" / "policy_cards" / "source_cards.json"
FROZEN_VERIFIER = ROOT / "data" / "policy_cards" / "frozen_learned_verifier_stage4.json"


class Stage4OrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._compiled = tempfile.TemporaryDirectory()
        compiled_root = Path(cls._compiled.name)
        compile_policy_cards(SOURCE_CARDS, compiled_root)
        cls.cards = load_compiled_policy_cards(compiled_root / "policy_cards.json")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._compiled.cleanup()

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temporary.name) / "stage4.sqlite3"
        self.store = SQLiteCheckpointStore(self.db_path)
        self.orchestrator = DurableStage4Orchestrator(self.store)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    @classmethod
    def _card(cls, card_id: str):
        return next(card for card in cls.cards if card.card_id == card_id)

    def _inputs(
        self,
        *,
        case_id: str = "STAGE4-SAFE-1",
        cause: str = "LATE_ALLOCATION",
        action: str = "REQUEST_ALLOCATION",
        card_id: str = "sec.rule15c6-2.aca.trade_date",
        confidence: float = 0.95,
        review_capacity: bool = True,
        planner_hold: bool = False,
    ) -> Stage4Inputs:
        request = PlannerRequest(
            case_id=case_id,
            decision_time="2026-01-15T20:00:00Z",
            predicted_cause=cause,
            candidate_action=action,
            permitted_actions=(action, "HOLD_FOR_REVIEW"),
        )
        if planner_hold:
            packet = hold_packet(
                request,
                "INSUFFICIENT_EVIDENCE",
                model_id="stage4-test",
            )
            hits: list[RetrievalHit] = []
        else:
            packet = ActionPacket(
                schema_version="1.0",
                case_id=case_id,
                decision_time=request.decision_time,
                root_cause=cause,
                action_code=action,
                evidence_ids=(card_id,),
                confidence=confidence,
                abstain=False,
                reason_code="SUPPORTED_ACTION",
                model_id="Qwen/Qwen3-4B",
                prompt_version="finops_action_packet_v1",
            )
            card = self._card(card_id)
            hits = [
                RetrievalHit(
                    document_id=card.card_id,
                    source_type="POLICY_CARD",
                    score=3.0,
                    rank=1,
                    allowed_actions=card.allowed_actions,
                    checksum=card.checksum,
                )
            ]
        return default_stage4_inputs(
            request=request,
            packet=packet,
            hits=hits,
            cards=self.cards,
            context=PolicyContext(
                requested_target="TRADEOPS_SIM",
                external_execution_requested=False,
                review_capacity_available=review_capacity,
                claim_boundary=CLAIM_BOUNDARY,
            ),
            verifier_path=FROZEN_VERIFIER,
        )

    def test_durable_path_matches_reference_and_records_effect_once(self) -> None:
        inputs = self._inputs()
        reference = run_stage4_reference(inputs)
        durable = self.orchestrator.start("safe-parity", inputs)
        self.assertEqual(
            durable.model_copy(update={"simulation_effect_recorded": False}),
            reference,
        )
        self.assertTrue(durable.simulation_effect_recorded)
        self.assertEqual(
            self.store.simulation_effect_count(durable.direct_state.idempotency_key),
            1,
        )
        replay = self.orchestrator.start("safe-parity", inputs)
        self.assertEqual(replay, durable)
        self.assertEqual(
            self.store.simulation_effect_count(durable.direct_state.idempotency_key),
            1,
        )

    def test_crash_after_effect_recovers_without_duplicate(self) -> None:
        inputs = self._inputs(case_id="STAGE4-CRASH-1")
        with self.assertRaisesRegex(InjectedCrash, "after idempotent effect"):
            self.orchestrator.start("crash-effect", inputs, crash_at="after_effect")
        reference_key = run_stage4_reference(inputs).direct_state.idempotency_key
        self.assertEqual(self.store.simulation_effect_count(reference_key), 1)
        recovered = DurableStage4Orchestrator(SQLiteCheckpointStore(self.db_path)).recover(
            "crash-effect"
        )
        self.assertTrue(recovered.simulation_effect_recorded)
        self.assertEqual(self.store.simulation_effect_count(reference_key), 1)

    def test_crash_after_input_recovers_from_durable_checkpoint(self) -> None:
        inputs = self._inputs(case_id="STAGE4-CRASH-INPUT")
        with self.assertRaisesRegex(InjectedCrash, "after input checkpoint"):
            self.orchestrator.start("crash-input", inputs, crash_at="after_input")
        recovered = DurableStage4Orchestrator(SQLiteCheckpointStore(self.db_path)).recover(
            "crash-input"
        )
        self.assertEqual(recovered.effective_route, "SIMULATE")
        self.assertTrue(recovered.simulation_effect_recorded)

    def test_database_is_immediately_removable_after_normal_operation(self) -> None:
        self.store.bind_thread("removable-normal", "a" * 64)
        self.assertEqual(self.store.load("removable-normal"), ())
        self.db_path.unlink()
        self.assertFalse(self.db_path.exists())

    def test_database_is_immediately_removable_after_injected_failure(self) -> None:
        def fail_receipt() -> dict[str, object]:
            raise RuntimeError("injected receipt failure")

        with self.assertRaisesRegex(RuntimeError, "injected receipt failure"):
            self.store.record_simulation_once(
                idempotency_key="b" * 64,
                request={"case_id": "failure"},
                receipt_factory=fail_receipt,
            )
        self.db_path.unlink()
        self.assertFalse(self.db_path.exists())

    def test_review_interrupt_resume_and_crash_replay_are_idempotent(self) -> None:
        inputs = self._inputs(
            case_id="STAGE4-REVIEW-1",
            cause="PRICE_MISMATCH",
            action="CORRECT_ECONOMIC_FIELD",
            card_id="sec.rule15c6-2.discrepancy_delays",
        )
        pending = self.orchestrator.start("review-crash", inputs)
        self.assertEqual(pending.direct_state.status, "WAITING_REVIEW")
        self.assertFalse(pending.simulation_effect_recorded)
        token = pending.direct_state.review_request.review_token
        with self.assertRaisesRegex(InjectedCrash, "after idempotent effect"):
            self.orchestrator.resume_review(
                "review-crash",
                review_token=token,
                reviewer_id="reviewer-1",
                decision="APPROVE",
                crash_at="after_effect",
            )
        recovered = DurableStage4Orchestrator(SQLiteCheckpointStore(self.db_path)).recover(
            "review-crash"
        )
        self.assertEqual(recovered.direct_state.status, "COMPLETED")
        self.assertEqual(recovered.effective_route, "SIMULATE")
        self.assertTrue(recovered.simulation_effect_recorded)
        self.assertEqual(
            self.store.simulation_effect_count(recovered.direct_state.idempotency_key),
            1,
        )
        replay = self.orchestrator.resume_review(
            "review-crash",
            review_token=token,
            reviewer_id="reviewer-1",
            decision="APPROVE",
        )
        self.assertEqual(replay, recovered)

    def test_conflicting_replay_is_rejected(self) -> None:
        inputs = self._inputs()
        self.orchestrator.start("conflict", inputs)
        conflicting = self._inputs(case_id="DIFFERENT-CASE")
        with self.assertRaisesRegex(
            CheckpointConflictError,
            "different inputs",
        ):
            self.orchestrator.start("conflict", conflicting)

    def test_learned_verifier_is_veto_only_and_flags_abstention(self) -> None:
        inputs = self._inputs(case_id="STAGE4-HOLD", planner_hold=True)
        state = self.orchestrator.start("learned-hold", inputs)
        self.assertTrue(state.learned_verifier.flagged)
        self.assertEqual(state.learned_verifier.mode, "VETO_ONLY")
        self.assertEqual(state.effective_route, "HOLD")
        self.assertFalse(state.simulation_effect_recorded)

    def test_release_freezes_checkpoint_history(self) -> None:
        inputs = self._inputs(case_id="STAGE4-RELEASE")
        state = self.orchestrator.start("release", inputs)
        release = self.orchestrator.release("release")
        self.assertEqual(release["status"], "RELEASED")
        verified = self.store.verify_release(
            "release",
            release["state_sha256"],
        )
        self.assertEqual(verified, release)
        with self.assertRaisesRegex(ReleasedTraceError, "immutable"):
            self.store.append(
                "release",
                "STATE",
                state.model_dump(mode="json"),
            )

    def test_checkpoint_hash_tamper_is_detected(self) -> None:
        self.orchestrator.start("tamper", self._inputs(case_id="STAGE4-TAMPER"))
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(
                """
                UPDATE checkpoints
                SET payload_json = '{"tampered":true}'
                WHERE thread_id = ? AND sequence = 1
                """,
                ("tamper",),
            )
        with self.assertRaisesRegex(CheckpointConflictError, "hash chain"):
            self.store.load("tamper")

    def test_optional_langgraph_adapter_has_explicit_dependency_boundary(self) -> None:
        try:
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            with self.assertRaisesRegex(
                LangGraphUnavailableError,
                r"\.\[orchestration\]",
            ):
                build_langgraph_stage4(object())
        else:
            graph = build_langgraph_stage4(InMemorySaver())
            self.assertIsNotNone(graph)

    def test_locked_core_qualification_publishes_verified_artifact(self) -> None:
        output = Path(self._temporary.name) / "qualification"
        result = run_stage4_qualification(
            source_cards_path=SOURCE_CARDS,
            scenarios_path=(
                ROOT / "data" / "policy_cards" / "orchestration_qualification_stage4_locked.json"
            ),
            verifier_path=FROZEN_VERIFIER,
            output_dir=output,
            require_langgraph=False,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["core_status"], "PASS")
        self.assertEqual(result["scenario_count"], 9)
        self.assertEqual(verify_stage4_qualification(output), result)

    def test_frozen_verifier_is_reproducible_and_provenance_bound(self) -> None:
        training = ROOT / "data" / "policy_cards" / "learned_verifier_training_stage4.json"
        threshold = ROOT / "data" / "policy_cards" / "learned_verifier_threshold_stage4.json"
        generated = Path(self._temporary.name) / "frozen.json"
        freeze(
            training_path=training,
            threshold_path=threshold,
            output_path=generated,
        )
        generated_bytes = generated.read_bytes()
        self.assertNotIn(b"\r\n", generated_bytes)
        self.assertEqual(generated_bytes, FROZEN_VERIFIER.read_bytes())
        artifact, _ = load_frozen_verifier(FROZEN_VERIFIER)
        result = verify_frozen_verifier_provenance(
            artifact,
            training_path=training,
            threshold_path=threshold,
        )
        self.assertEqual(result["status"], "PASS")
