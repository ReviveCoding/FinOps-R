from __future__ import annotations

import unittest

import pandas as pd

from finops_r.data import dataframe_fingerprint, generate_cases, validate_cases


class DataTests(unittest.TestCase):
    def test_generation_is_deterministic_and_entity_safe(self) -> None:
        first = generate_cases(700, 350, 17)
        second = generate_cases(700, 350, 17)
        self.assertEqual(dataframe_fingerprint(first), dataframe_fingerprint(second))
        report = validate_cases(first)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["entity_split_leakage_count"], 0)
        self.assertEqual(set(first["root_cause"].unique()) - set(first["root_cause"]), set())

    def test_fingerprint_is_invariant_to_out_of_order_rows(self) -> None:
        frame = generate_cases(700, 350, 17)
        shuffled = frame.sample(frac=1.0, random_state=99).reset_index(drop=True)
        self.assertEqual(dataframe_fingerprint(frame), dataframe_fingerprint(shuffled))
        self.assertEqual(validate_cases(shuffled)["status"], "PASS")

    def test_invalid_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_cases(50, 350, 17)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            generate_cases(200, 201, 17)

    def test_empty_and_malformed_frames_fail_without_crashing(self) -> None:
        empty = validate_cases(pd.DataFrame())
        malformed = validate_cases(pd.DataFrame([{"case_id": "T1", "state": "EXECUTED"}]))
        self.assertEqual(empty["status"], "FAIL")
        self.assertEqual(malformed["status"], "FAIL")

    def test_full_contract_detects_corruption_without_stopping_at_first_error(self) -> None:
        frame = generate_cases(700, 350, 17)
        frame.loc[0, "case_id"] = " "
        frame.loc[1, "case_id"] = frame.loc[2, "case_id"]
        frame.loc[3, "decision_time"] = "2027-01-01T00:00:00"
        frame.loc[4, "allocation_present"] = 0.5
        frame.loc[5, "harm"] = -1.0
        frame.loc[6, "oracle_action"] = "WRONG"
        frame.loc[7, "root_cause"] = "UNKNOWN_CAUSE"
        frame.loc[8, "split"] = "future"
        report = validate_cases(frame)
        self.assertEqual(report["status"], "FAIL")
        for key in (
            "invalid_identifier_count",
            "duplicate_case_count",
            "invalid_decision_time_count",
            "invalid_binary_count",
            "negative_numeric_count",
            "invalid_oracle_action_count",
            "invalid_cause_count",
            "invalid_split_count",
        ):
            self.assertGreater(report[key], 0, key)

    def test_duplicate_columns_and_non_finite_values_fail_cleanly(self) -> None:
        frame = generate_cases(700, 350, 17)
        frame.loc[0, "harm"] = float("inf")
        frame.insert(len(frame.columns), "state_copy", frame["state"])
        frame.columns = [*frame.columns[:-1], "state"]
        report = validate_cases(frame)
        self.assertEqual(report["status"], "FAIL")
        self.assertGreater(report["duplicate_column_count"], 0)
        self.assertGreater(report["non_finite_numeric_count"], 0)

    def test_fractional_count_fields_fail_the_data_contract(self) -> None:
        frame = generate_cases(700, 350, 17)
        frame["event_count"] = frame["event_count"].astype(float)
        frame.loc[0, "event_count"] = 1.5
        report = validate_cases(frame)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["invalid_integer_count"], 1)

    def test_missing_splits_and_training_labels_fail_before_model_fitting(self) -> None:
        frame = generate_cases(700, 350, 17)
        one_split = frame.copy()
        one_split["split"] = "train"
        report = validate_cases(one_split)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["missing_split_count"], 5)

        missing_label = frame.copy()
        cause = str(
            missing_label.loc[
                (missing_label["split"] == "train")
                & (missing_label["root_cause"] != "NO_EXCEPTION"),
                "root_cause",
            ].iloc[0]
        )
        mask = (missing_label["split"] == "train") & (missing_label["root_cause"] == cause)
        missing_label.loc[mask, "root_cause"] = "NO_EXCEPTION"
        missing_label.loc[mask, "oracle_action"] = "NO_ACTION_MONITOR"
        missing_label.loc[mask, "harm"] = 1.0
        missing_label.loc[mask, "critical"] = 0
        report = validate_cases(missing_label)
        self.assertEqual(report["status"], "FAIL")
        self.assertGreater(report["training_missing_cause_count"], 0)


if __name__ == "__main__":
    unittest.main()
