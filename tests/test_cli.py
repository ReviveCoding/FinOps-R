from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from finops_r import __version__
from finops_r.cli import _safe_run_id, main


class CliTests(unittest.TestCase):
    def test_safe_run_id_accepts_canonical_identifier(self) -> None:
        value = "finopsr_20260718T120000Z_0123456789"
        self.assertEqual(_safe_run_id(value), value)

    def test_run_id_rejects_traversal_absolute_hidden_and_windows_paths(self) -> None:
        for value in (
            "../run",
            "/tmp/run",
            "C:\\tmp\\run",
            ".hidden",
            "run/name",
            " run",
            "run.json",
            "CON",
            "com1",
            "x" * 129,
            "",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _safe_run_id(value)

    def test_version_is_available_from_public_entrypoint(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"finops-r {__version__}")

    def test_default_error_is_safe_and_debug_preserves_exception(self) -> None:
        stderr = io.StringIO()
        with patch("finops_r.cli.run_project", side_effect=RuntimeError("injected failure")):
            with contextlib.redirect_stderr(stderr):
                exit_code = main(["run", "--config", "missing.json"])
        self.assertEqual(exit_code, 2)
        self.assertIn('"status": "ERROR"', stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

        with patch("finops_r.cli.run_project", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                main(["--debug", "run", "--config", "missing.json"])

    def test_keyboard_interrupt_uses_shell_standard_exit_without_traceback(self) -> None:
        stderr = io.StringIO()
        with patch("finops_r.cli.run_project", side_effect=KeyboardInterrupt):
            with contextlib.redirect_stderr(stderr):
                exit_code = main(["run", "--config", "missing.json"])
        self.assertEqual(exit_code, 130)
        self.assertIn('"type": "KeyboardInterrupt"', stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

        with patch("finops_r.cli.run_project", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                main(["--debug", "run", "--config", "missing.json"])


if __name__ == "__main__":
    unittest.main()
