from __future__ import annotations

import unittest

from finops_r.path_safety import is_portable_component


class PathSafetyTests(unittest.TestCase):
    def test_unpaired_surrogate_is_rejected_without_encoding_error(self) -> None:
        self.assertFalse(is_portable_component("\ud800"))


if __name__ == "__main__":
    unittest.main()
