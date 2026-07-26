from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from finops_r.orchestration_stage3 import (
    resume_human_review,
    run_direct_orchestrator,
    trace_contract_is_complete,
)
from finops_r.planner import ActionPacket, PlannerRequest, hold_packet
from finops_r.policy_cards import (
    CompiledPolicyCard,
    compile_policy_cards,
    load_compiled_policy_cards,
)
from finops_r.retrieval import RetrievalHit
from finops_r.verification_policy import (
    CLAIM_BOUNDARY,
    SIMULATION_TARGET,
    PolicyContext,
    apply_simulation_policy,
    verify_action_packet,
)


ROOT = Path(__file__).parents[1]
SOURCE_CARDS = ROOT / "data" / "policy_cards" / "source_cards.json"


class VerificationPolicyOrchestrationStage3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        compiled_root = Path(cls._temporary.name)
        compile_policy_cards(SOURCE_CARDS, compiled_root)
        cls.cards = load_compiled_policy_cards(compiled_root / "policy_cards.json")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _card(self, card_id: str) -> CompiledPolicyCard:
        return next(card for card in self.cards if card.card_id == card_id)

    def _request(self, **overrides: object) -> PlannerRequest:
        payload: dict[str, object] = {
            "case_id": "STAGE3-CASE-1",
            "decision_time": "2026-01-15T20:00:00Z",
            "predicted_cause": "LATE_ALLOCATION",
            "candidate_action": "REQUEST_ALLOCATION",
            "permitted_actions": ("REQUEST_ALLOCATION", "HOLD_FOR_REVIEW"),
        }
        payload.update(overrides)
        return PlannerRequest.model_validate(payload)

    def _packet(self, request: PlannerRequest, **overrides: object) -> ActionPacket:
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "case_id": request.case_id,
            "decision_time": request.decision_time,
            "root_cause": request.predicted_cause,
            "action_code": request.candidate_action,
            "evidence_ids": ("sec.rule15c6-2.aca.trade_date",),
            "confidence": 0.95,
            "abstain": False,
            "reason_code": "SUPPORTED_ACTION",
            "model_id": "Qwen/Qwen3-4B",
            "prompt_version": "finops_action_packet_v1",
        }
        payload.update(overrides)
        return ActionPacket.model_validate(payload)

    def _hit(self, card_id: str, **overrides: object) -> RetrievalHit:
        card = self._card(card_id)
        payload: dict[str, object] = {
            "document_id": card.card_id,
            "source_type": "POLICY_CARD",
            "score": 3.0,
            "rank": 1,
            "allowed_actions": card.allowed_actions,
            "checksum": card.checksum,
        }
        payload.update(overrides)
        return RetrievalHit.model_validate(payload)

    def _context(self, **overrides: object) -> PolicyContext:
        payload: dict[str, object] = {
            "requested_target": SIMULATION_TARGET,
            "external_execution_requested": False,
            "review_capacity_available": True,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        payload.update(overrides)
        return PolicyContext.model_validate(payload)

    def test_supported_low_risk_action_is_simulated_only(self) -> None:
        request = self._request()
        packet = self._packet(request)
        state = run_direct_orchestrator(
            request,
            packet,
            [self._hit("sec.rule15c6-2.aca.trade_date")],
            self.cards,
            self._context(),
        )
        self.assertEqual(state.verification.status, "SUPPORTED")
        self.assertEqual(state.route, "SIMULATE")
        self.assertEqual(state.status, "COMPLETED")
        self.assertIsNotNone(state.receipt)
        self.assertEqual(state.receipt.target, "TRADEOPS_SIM")
        self.assertFalse(state.receipt.external_execution)
        self.assertTrue(state.trace_complete)
        self.assertEqual(
            [event.node for event in state.trace],
            ["INGEST", "VERIFY", "POLICY", "ROUTE", "SIMULATE", "COMPLETE"],
        )

    def test_correction_requires_review_and_approval_is_idempotent(self) -> None:
        request = self._request(
            predicted_cause="PRICE_MISMATCH",
            candidate_action="CORRECT_ECONOMIC_FIELD",
            permitted_actions=("CORRECT_ECONOMIC_FIELD", "HOLD_FOR_REVIEW"),
        )
        packet = self._packet(
            request,
            action_code="CORRECT_ECONOMIC_FIELD",
            evidence_ids=("sec.rule15c6-2.discrepancy_delays",),
        )
        pending = run_direct_orchestrator(
            request,
            packet,
            [self._hit("sec.rule15c6-2.discrepancy_delays")],
            self.cards,
            self._context(),
        )
        self.assertEqual(pending.status, "WAITING_REVIEW")
        self.assertEqual(pending.route, "REVIEW")
        self.assertIsNotNone(pending.review_request)
        approved = resume_human_review(
            pending,
            review_token=pending.review_request.review_token,
            reviewer_id="ops-reviewer-7",
            decision="APPROVE",
        )
        replayed = resume_human_review(
            approved,
            review_token=pending.review_request.review_token,
            reviewer_id="ops-reviewer-7",
            decision="APPROVE",
        )
        self.assertEqual(approved, replayed)
        self.assertEqual(approved.route, "SIMULATE")
        self.assertEqual(approved.receipt.idempotency_key, pending.idempotency_key)
        self.assertFalse(approved.receipt.external_execution)

    def test_review_rejection_holds_and_conflicting_replay_is_rejected(self) -> None:
        request = self._request(
            predicted_cause="PRICE_MISMATCH",
            candidate_action="CORRECT_ECONOMIC_FIELD",
            permitted_actions=("CORRECT_ECONOMIC_FIELD", "HOLD_FOR_REVIEW"),
        )
        packet = self._packet(
            request,
            action_code="CORRECT_ECONOMIC_FIELD",
            evidence_ids=("sec.rule15c6-2.discrepancy_delays",),
        )
        pending = run_direct_orchestrator(
            request,
            packet,
            [self._hit("sec.rule15c6-2.discrepancy_delays")],
            self.cards,
            self._context(),
        )
        rejected = resume_human_review(
            pending,
            review_token=pending.review_request.review_token,
            reviewer_id="ops-reviewer-8",
            decision="REJECT",
        )
        self.assertEqual(rejected.route, "HOLD")
        self.assertIsNone(rejected.receipt)
        with self.assertRaisesRegex(ValueError, "already been resolved"):
            resume_human_review(
                rejected,
                review_token=pending.review_request.review_token,
                reviewer_id="ops-reviewer-8",
                decision="APPROVE",
            )

    def test_review_capacity_overflow_fails_closed(self) -> None:
        request = self._request(
            predicted_cause="PRICE_MISMATCH",
            candidate_action="CORRECT_ECONOMIC_FIELD",
            permitted_actions=("CORRECT_ECONOMIC_FIELD", "HOLD_FOR_REVIEW"),
        )
        packet = self._packet(
            request,
            action_code="CORRECT_ECONOMIC_FIELD",
            evidence_ids=("sec.rule15c6-2.discrepancy_delays",),
        )
        state = run_direct_orchestrator(
            request,
            packet,
            [self._hit("sec.rule15c6-2.discrepancy_delays")],
            self.cards,
            self._context(review_capacity_available=False),
        )
        self.assertEqual(state.route, "HOLD")
        self.assertEqual(state.policy.reason_code, "REVIEW_CAPACITY_UNAVAILABLE")
        self.assertIsNone(state.receipt)

    def test_external_target_and_external_execution_are_prohibited(self) -> None:
        request = self._request()
        packet = self._packet(request)
        verification = verify_action_packet(
            packet,
            request,
            [self._hit("sec.rule15c6-2.aca.trade_date")],
            self.cards,
        )
        decision = apply_simulation_policy(
            packet,
            verification,
            self._context(
                requested_target="BROKER_API",
                external_execution_requested=True,
            ),
        )
        self.assertEqual(decision.route, "HOLD")
        self.assertIn("POLICY_EXTERNAL_TARGET_PROHIBITED", decision.violation_rule_ids)
        self.assertIn("POLICY_EXTERNAL_EXECUTION_PROHIBITED", decision.violation_rule_ids)

    def test_wrong_claim_boundary_is_prohibited(self) -> None:
        request = self._request()
        packet = self._packet(request)
        verification = verify_action_packet(
            packet,
            request,
            [self._hit("sec.rule15c6-2.aca.trade_date")],
            self.cards,
        )
        decision = apply_simulation_policy(
            packet,
            verification,
            self._context(claim_boundary="BANK_PRODUCTION"),
        )
        self.assertEqual(decision.route, "HOLD")
        self.assertIn("POLICY_CLAIM_BOUNDARY_MISMATCH", decision.violation_rule_ids)

    def test_unretrieved_and_non_policy_citations_are_rejected(self) -> None:
        request = self._request()
        packet = self._packet(request)
        missing = verify_action_packet(packet, request, [], self.cards)
        self.assertEqual(missing.status, "REJECTED")
        self.assertIn("VERIFIER_UNRETRIEVED_EVIDENCE", missing.violation_rule_ids)
        non_policy = verify_action_packet(
            packet,
            request,
            [
                self._hit(
                    "sec.rule15c6-2.aca.trade_date",
                    source_type="CASE_ARTIFACT",
                )
            ],
            self.cards,
        )
        self.assertIn("VERIFIER_NON_POLICY_EVIDENCE", non_policy.violation_rule_ids)

    def test_tampered_card_and_hit_checksum_are_rejected(self) -> None:
        request = self._request()
        packet = self._packet(request)
        card = self._card("sec.rule15c6-2.aca.trade_date")
        tampered = card.model_copy(update={"checksum": "0" * 64})
        result = verify_action_packet(
            packet,
            request,
            [self._hit(card.card_id, checksum="0" * 64)],
            [tampered],
        )
        self.assertIn("VERIFIER_POLICY_CARD_CHECKSUM_MISMATCH", result.violation_rule_ids)
        wrong_hit = verify_action_packet(
            packet,
            request,
            [self._hit(card.card_id, checksum="f" * 64)],
            self.cards,
        )
        self.assertIn(
            "VERIFIER_RETRIEVAL_CARD_CHECKSUM_MISMATCH",
            wrong_hit.violation_rule_ids,
        )

    def test_expired_policy_card_is_rejected(self) -> None:
        request = self._request(
            predicted_cause="NO_EXCEPTION",
            candidate_action="NO_ACTION_MONITOR",
            permitted_actions=("NO_ACTION_MONITOR", "HOLD_FOR_REVIEW"),
        )
        packet = self._packet(
            request,
            action_code="NO_ACTION_MONITOR",
            evidence_ids=("sec.rule15c6-1.standard_cycle.tplus2.legacy",),
        )
        result = verify_action_packet(
            packet,
            request,
            [self._hit("sec.rule15c6-1.standard_cycle.tplus2.legacy")],
            self.cards,
        )
        self.assertIn("VERIFIER_POLICY_CARD_NOT_EFFECTIVE", result.violation_rule_ids)

    def test_packet_request_binding_mismatches_are_rejected(self) -> None:
        request = self._request()
        packet = self._packet(
            request,
            case_id="OTHER-CASE",
            decision_time="2026-01-15T20:01:00Z",
            root_cause="MISSING_AFFIRMATION",
        )
        result = verify_action_packet(
            packet,
            request,
            [self._hit("sec.rule15c6-2.aca.trade_date")],
            self.cards,
        )
        self.assertIn("VERIFIER_CASE_ID_MISMATCH", result.violation_rule_ids)
        self.assertIn("VERIFIER_DECISION_TIME_MISMATCH", result.violation_rule_ids)
        self.assertIn("VERIFIER_ROOT_CAUSE_MISMATCH", result.violation_rule_ids)

    def test_unsupported_extra_citation_rejects_the_whole_packet(self) -> None:
        request = self._request()
        packet = self._packet(
            request,
            evidence_ids=(
                "sec.rule15c6-2.aca.trade_date",
                "sec.rule15c6-1.standard_cycle.tplus1",
            ),
        )
        result = verify_action_packet(
            packet,
            request,
            [
                self._hit("sec.rule15c6-2.aca.trade_date", rank=1),
                self._hit("sec.rule15c6-1.standard_cycle.tplus1", rank=2),
            ],
            self.cards,
        )
        self.assertEqual(result.status, "REJECTED")
        self.assertIn("VERIFIER_CITATION_DOES_NOT_SUPPORT_ACTION", result.violation_rule_ids)
        self.assertIn("VERIFIER_INCOMPLETE_CITATION_SUPPORT", result.violation_rule_ids)

    def test_explicit_planner_hold_remains_held(self) -> None:
        request = self._request()
        packet = hold_packet(request, "INSUFFICIENT_EVIDENCE", model_id="Qwen/Qwen3-4B")
        state = run_direct_orchestrator(request, packet, [], self.cards, self._context())
        self.assertEqual(state.verification.status, "ABSTAINED")
        self.assertEqual(state.route, "HOLD")
        self.assertEqual(state.policy.reason_code, "PLANNER_ABSTENTION")
        self.assertTrue(state.trace_complete)

    def test_trace_hashes_and_sequences_are_complete(self) -> None:
        request = self._request()
        packet = self._packet(request)
        first = run_direct_orchestrator(
            request,
            packet,
            [self._hit("sec.rule15c6-2.aca.trade_date")],
            self.cards,
            self._context(),
        )
        second = run_direct_orchestrator(
            request,
            packet,
            [self._hit("sec.rule15c6-2.aca.trade_date")],
            self.cards,
            self._context(),
        )
        self.assertEqual(first, second)
        self.assertEqual([event.sequence for event in first.trace], [1, 2, 3, 4, 5, 6])
        self.assertTrue(
            all(
                len(event.input_sha256) == 64 and len(event.output_sha256) == 64
                for event in first.trace
            )
        )
        self.assertTrue(trace_contract_is_complete(first))
        missing_complete = first.model_copy(update={"trace": first.trace[:-1]})
        self.assertFalse(trace_contract_is_complete(missing_complete))


if __name__ == "__main__":
    unittest.main()
