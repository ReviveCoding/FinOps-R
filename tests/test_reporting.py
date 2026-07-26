from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ReportingEnvironmentTests(unittest.TestCase):
    def test_default_matplotlib_cache_uses_platform_temp_directory(self) -> None:
        environment = os.environ.copy()
        environment.pop("MPLCONFIGDIR", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, os, tempfile; "
                    "import finops_r.reporting; "
                    "print(json.dumps({'actual': os.environ['MPLCONFIGDIR'], "
                    "'expected': str(__import__('pathlib').Path(tempfile.gettempdir()) / "
                    "'finops-r-matplotlib')}))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        value = json.loads(result.stdout)
        self.assertEqual(Path(value["actual"]), Path(value["expected"]))

    def test_existing_matplotlib_cache_override_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = directory
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os; import finops_r.reporting; print(os.environ['MPLCONFIGDIR'])",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(Path(result.stdout.strip()), Path(directory))


if __name__ == "__main__":
    unittest.main()
