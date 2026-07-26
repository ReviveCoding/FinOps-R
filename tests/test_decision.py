from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from finops_r.decision import fit_review_threshold, route_decisions


def _frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [f"T{index}" for index in range(rows)],
            "evidence_complete": np.ones(rows),
            "ssi_version_gap": np.zeros(rows),
            "qty_diff_pct": np.zeros(rows),
            "price_diff_bps": np.zeros(rows),
            "funding_ratio": np.ones(rows),
            "securities_ratio": np.ones(rows),
            "compliance_score": np.zeros(rows),
        }
    )


class DecisionTests(unittest.TestCase):
    def test_hard_holds_consume_the_fixed_manual_capacity(self) -> None:
        rows = 10
        frame = _frame(rows)
        probabilities = np.tile(np.asarray([[0.55, 0.45]]), (rows, 1))
        predictions = np.asarray(["NO_EXCEPTION"] * rows)
        envelopes = pd.DataFrame(
            {
                "policy_status": ["BLOCK", "BLOCK"] + ["PASS"] * 8,
                "verifier_status": ["INSUFFICIENT", "INSUFFICIENT"] + ["SUPPORTED"] * 8,
            }
        )
        routes = route_decisions(
            frame,
            predictions,
            probabilities,
            policy="confidence",
            review_capacity=0.4,
            human_correction_probability=0.85,
            review_cost=0.1,
            envelopes=envelopes,
        )
        counts = routes["route"].value_counts().to_dict()
        self.assertEqual(counts["HOLD"], 2)
        self.assertEqual(counts["REVIEW"], 2)
        self.assertEqual(int((routes["route"] != "SIMULATE").sum()), 4)

    def test_evr_does_not_fill_capacity_with_negative_value_reviews(self) -> None:
        rows = 10
        frame = _frame(rows)
        probabilities = np.tile(np.asarray([[0.999, 0.001]]), (rows, 1))
        predictions = np.asarray(["NO_EXCEPTION"] * rows)
        routes = route_decisions(
            frame,
            predictions,
            probabilities,
            policy="evr",
            review_capacity=0.5,
            human_correction_probability=0.85,
            review_cost=1.0,
        )
        self.assertEqual(int((routes["route"] == "REVIEW").sum()), 0)

    def test_review_threshold_is_fit_on_policy_data_and_reused(self) -> None:
        rows = 10
        frame = _frame(rows)
        error_risk = np.linspace(0.01, 0.50, rows)
        probabilities = np.column_stack([1.0 - error_risk, error_risk])
        predictions = np.asarray(["NO_EXCEPTION"] * rows)
        threshold = fit_review_threshold(
            frame, predictions, probabilities, "confidence", 0.2, 0.85, 0.1
        )
        self.assertAlmostEqual(threshold, error_risk[-2])
        shifted = np.column_stack(
            [1.0 - np.linspace(0.02, 0.60, rows), np.linspace(0.02, 0.60, rows)]
        )
        routed = route_decisions(
            frame,
            predictions,
            shifted,
            "confidence",
            0.5,
            0.85,
            0.1,
            minimum_review_score=threshold,
        )
        reviewed = routed.loc[routed["route"] == "REVIEW", "review_score"]
        self.assertTrue((reviewed >= threshold).all())
        self.assertLessEqual(len(reviewed), 5)

    def test_threshold_replay_cannot_exceed_capacity_on_boundary_ties(self) -> None:
        rows = 10
        frame = _frame(rows)
        probabilities = np.tile(np.asarray([[0.6, 0.4]]), (rows, 1))
        predictions = np.asarray(["NO_EXCEPTION"] * rows)
        threshold = fit_review_threshold(
            frame, predictions, probabilities, "confidence", 0.2, 0.85, 0.1
        )
        replayed_reviews = 0
        for index in range(rows):
            routed = route_decisions(
                frame.iloc[[index]],
                predictions[[index]],
                probabilities[[index]],
                "confidence",
                1.0,
                0.85,
                0.1,
                minimum_review_score=threshold,
            )
            replayed_reviews += int(routed.iloc[0]["route"] == "REVIEW")
        self.assertLessEqual(replayed_reviews, 2)

    def test_invalid_probability_and_disagreement_contracts_are_rejected(self) -> None:
        frame = _frame(2)
        predictions = np.asarray(["NO_EXCEPTION", "NO_EXCEPTION"])
        invalid_inputs = (
            (np.asarray([[0.5, 0.4], [0.5, 0.5]]), None),
            (np.asarray([[0.5, 0.5], [0.5, float("nan")]]), None),
            (np.asarray([[0.5, 0.5], [0.5, 0.5]]), np.asarray([-0.1, 0.2])),
        )
        for probabilities, disagreement in invalid_inputs:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                route_decisions(
                    frame,
                    predictions,
                    probabilities,
                    "disagreement",
                    0.5,
                    0.85,
                    0.1,
                    disagreement=disagreement,
                )

    def test_unknown_cause_is_rejected_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown predicted cause"):
            route_decisions(
                _frame(1),
                np.asarray(["UNKNOWN"]),
                np.asarray([[0.5, 0.5]]),
                "confidence",
                0.2,
                0.85,
                0.1,
            )


if __name__ == "__main__":
    unittest.main()
