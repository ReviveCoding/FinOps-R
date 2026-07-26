from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from finops_r.evaluation import (
    benchmark_latency,
    paired_bootstrap_delta,
    paired_bootstrap_macro_f1_delta,
    predictive_metrics,
)


class EvaluationContractTests(unittest.TestCase):
    def test_single_class_slice_metrics_remain_finite(self) -> None:
        truth = np.asarray(["A"] * 9)
        predictions = np.asarray(["A"] * 9)
        probabilities = np.tile(np.asarray([[0.95, 0.05]]), (9, 1))
        metrics = predictive_metrics(truth, predictions, probabilities, np.asarray(["A", "B"]))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_empty_latency_and_bootstrap_inputs_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one case"):
            benchmark_latency(lambda frame: np.zeros(len(frame)), pd.DataFrame())
        with self.assertRaisesRegex(ValueError, "requires data"):
            paired_bootstrap_delta(np.asarray([]), np.asarray([]), seed=1, resamples=100)
        with self.assertRaisesRegex(ValueError, "requires data"):
            paired_bootstrap_macro_f1_delta(
                np.asarray([]), np.asarray([]), np.asarray([]), seed=1, resamples=100
            )

    def test_macro_f1_bootstrap_keeps_the_complete_label_universe(self) -> None:
        truth = np.asarray(["A"] * 9 + ["B"])
        baseline = np.asarray(["A"] * 10)
        candidate = np.asarray(["A"] * 8 + ["B", "B"])
        seed = 37
        resamples = 250
        result = paired_bootstrap_macro_f1_delta(
            truth,
            baseline,
            candidate,
            seed=seed,
            resamples=resamples,
            classes=np.asarray(["A", "B"]),
        )

        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(resamples):
            sample = rng.integers(0, len(truth), size=len(truth))
            draws.append(
                f1_score(
                    truth[sample],
                    candidate[sample],
                    labels=["A", "B"],
                    average="macro",
                    zero_division=0,
                )
                - f1_score(
                    truth[sample],
                    baseline[sample],
                    labels=["A", "B"],
                    average="macro",
                    zero_division=0,
                )
            )
        self.assertAlmostEqual(result["ci95_lower"], float(np.percentile(draws, 2.5)))
        self.assertAlmostEqual(result["ci95_upper"], float(np.percentile(draws, 97.5)))
        self.assertEqual(result["label_count"], 2)
        self.assertEqual(result["resampling_unit"], "case")

    def test_group_bootstrap_resamples_complete_entities_deterministically(self) -> None:
        baseline = np.asarray([1.0, 2.0, 10.0, 20.0])
        candidate = np.zeros(4)
        groups = np.asarray(["E1", "E1", "E2", "E2"])
        seed = 11
        resamples = 200
        result = paired_bootstrap_delta(
            baseline, candidate, seed=seed, resamples=resamples, groups=groups
        )

        rng = np.random.default_rng(seed)
        clusters = (np.asarray([0, 1]), np.asarray([2, 3]))
        draws = []
        for _ in range(resamples):
            selected = rng.integers(0, 2, size=2)
            sample = np.concatenate([clusters[int(index)] for index in selected])
            draws.append(float(baseline[sample].mean()))
        self.assertAlmostEqual(result["ci95_lower"], float(np.percentile(draws, 2.5)))
        self.assertAlmostEqual(result["ci95_upper"], float(np.percentile(draws, 97.5)))
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["resampling_unit"], "group")

    def test_bootstrap_rejects_nonfinite_costs_invalid_groups_and_unknown_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_bootstrap_delta(
                np.asarray([1.0, np.nan]), np.asarray([0.0, 0.0]), seed=1, resamples=10
            )
        with self.assertRaisesRegex(ValueError, "one value per row"):
            paired_bootstrap_delta(
                np.asarray([1.0, 2.0]),
                np.asarray([0.0, 0.0]),
                seed=1,
                resamples=10,
                groups=np.asarray(["E1"]),
            )
        with self.assertRaisesRegex(ValueError, "outside classes"):
            paired_bootstrap_macro_f1_delta(
                np.asarray(["A", "B"]),
                np.asarray(["A", "A"]),
                np.asarray(["A", "B"]),
                seed=1,
                resamples=10,
                classes=np.asarray(["A"]),
            )


if __name__ == "__main__":
    unittest.main()
