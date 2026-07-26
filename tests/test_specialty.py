from __future__ import annotations

import unittest

from finops_r.data import generate_cases
from finops_r.specialty import RSOPTransformer


class RSOPTests(unittest.TestCase):
    def test_rsop_is_train_fitted_and_bounded(self) -> None:
        data = generate_cases(1200, 600, 31)
        train = data[data["split"] == "train"].reset_index(drop=True)
        test = data[data["split"] == "test"].reset_index(drop=True)
        rsop = RSOPTransformer(min_reference_count=6).fit(train, train["root_cause"])
        transformed = rsop.transform(test)
        self.assertTrue(
            ((transformed["rsop_percentile"] > 0) & (transformed["rsop_percentile"] < 1)).all()
        )
        self.assertTrue((transformed["delay_pressure"] >= 0).all())
        self.assertIn("group_reference_count", rsop.manifest())

    def test_transform_does_not_mutate_reference(self) -> None:
        data = generate_cases(1000, 500, 33)
        train = data[data["split"] == "train"].reset_index(drop=True)
        test = data[data["split"] == "test"].reset_index(drop=True)
        rsop = RSOPTransformer(min_reference_count=5).fit(train, train["root_cause"])
        before = rsop.manifest().copy()
        rsop.transform(test)
        self.assertEqual(before, rsop.manifest())


if __name__ == "__main__":
    unittest.main()
