from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from finops_r.constants import CAUSES
from finops_r.decision import route_decisions
from finops_r.evidence import build_action_envelopes


class EvidenceTests(unittest.TestCase):
    def _row(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "case_id": "T1",
                    "state": "INSTRUCTED",
                    "time_in_state_min": 100.0,
                    "evidence_complete": 1.0,
                    "ssi_version_gap": 2.0,
                    "ssi_present": 1.0,
                    "currency_match": 1.0,
                    "counterparty_match": 1.0,
                    "security_id_match": 1.0,
                    "allocation_present": 1.0,
                    "affirmation_present": 1.0,
                    "funding_ratio": 1.05,
                    "securities_ratio": 1.04,
                    "compliance_score": 0.1,
                    "qty_diff_pct": 0.0,
                    "price_diff_bps": 0.0,
                    "duplicate_amendment_count": 0.0,
                }
            ]
        )

    def test_aee_expands_supported_action_and_prunes(self) -> None:
        frame = self._row()
        probabilities = np.full((1, len(CAUSES)), 0.01 / (len(CAUSES) - 1))
        probabilities[0, CAUSES.index("NO_EXCEPTION")] = 0.99
        config = {
            "action_envelope": {
                "cumulative_probability": 0.8,
                "max_candidates": 3,
                "rule_expand_threshold": 0.65,
            }
        }
        envelope = build_action_envelopes(frame, probabilities, np.asarray(CAUSES), config).iloc[0]
        self.assertIn("REQUEST_SSI_REFRESH", envelope["expanded_actions"])
        self.assertIn("REQUEST_SSI_REFRESH", envelope["pruned_actions"])

    def test_missing_evidence_cannot_increase_automation(self) -> None:
        frame = self._row()
        probabilities = np.full((1, len(CAUSES)), 0.01 / (len(CAUSES) - 1))
        probabilities[0, CAUSES.index("STALE_SSI")] = 0.99
        config = {
            "action_envelope": {
                "cumulative_probability": 0.8,
                "max_candidates": 3,
                "rule_expand_threshold": 0.65,
            }
        }
        complete = build_action_envelopes(frame, probabilities, np.asarray(CAUSES), config)
        incomplete_frame = frame.copy()
        incomplete_frame["evidence_complete"] = 0.0
        incomplete = build_action_envelopes(
            incomplete_frame, probabilities, np.asarray(CAUSES), config
        )
        first = route_decisions(
            frame,
            np.asarray(["STALE_SSI"]),
            probabilities,
            "evr",
            0.0,
            0.85,
            0.1,
            envelopes=complete,
        )
        second = route_decisions(
            incomplete_frame,
            np.asarray(["STALE_SSI"]),
            probabilities,
            "evr",
            0.0,
            0.85,
            0.1,
            envelopes=incomplete,
        )
        self.assertEqual(first.iloc[0]["route"], "SIMULATE")
        self.assertEqual(second.iloc[0]["route"], "HOLD")


if __name__ == "__main__":
    unittest.main()
