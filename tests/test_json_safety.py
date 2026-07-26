from __future__ import annotations

import io
import math
import tempfile
import unittest
from pathlib import Path

from finops_r.api import MAX_ASSESSMENT_JSON_BYTES, _strict_request_object
from finops_r.artifacts import config_hash
from finops_r.datasets import DataContractError, load_dataset_registry
from finops_r.json_safety import (
    StrictJSONError,
    require_json_object,
    strict_json_dump,
    strict_json_dumps,
    strict_json_load,
    strict_json_loads,
)
from finops_r.pipeline import assess_request, load_config


class StrictJSONTests(unittest.TestCase):
    def test_valid_document_round_trips(self) -> None:
        value = {"name": "finops", "values": [1, 2.5, None, True]}
        self.assertEqual(strict_json_loads(strict_json_dumps(value)), value)

    def test_duplicate_keys_are_rejected_at_every_depth(self) -> None:
        with self.assertRaisesRegex(StrictJSONError, "duplicate JSON object key"):
            strict_json_loads('{"outer": {"id": 1, "id": 2}}')

    def test_nonstandard_constants_are_rejected(self) -> None:
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token), self.assertRaises(StrictJSONError):
                strict_json_loads('{"value": ' + token + "}")

    def test_float_overflow_is_rejected(self) -> None:
        with self.assertRaisesRegex(StrictJSONError, "finite float range"):
            strict_json_loads('{"value": 1e10000}')

    def test_nonfinite_python_values_are_rejected_on_write(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(StrictJSONError):
                strict_json_dumps({"value": value})

    def test_stream_helpers_and_object_requirement(self) -> None:
        handle = io.StringIO()
        strict_json_dump({"status": "PASS"}, handle, sort_keys=True)
        handle.seek(0)
        self.assertEqual(require_json_object(strict_json_load(handle)), {"status": "PASS"})
        with self.assertRaisesRegex(StrictJSONError, "top-level JSON object"):
            require_json_object([])

    def test_config_hash_rejects_nonfinite_values(self) -> None:
        with self.assertRaises(StrictJSONError):
            config_hash({"unsafe": math.nan})

    def test_registry_rejects_ambiguous_and_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            for document in (
                '{"schema_version":"1.0","schema_version":"2.0","datasets":[]}',
                '{"schema_version":"1.0","datasets":[],"score":NaN}',
            ):
                path.write_text(document, encoding="utf-8")
                with self.subTest(document=document), self.assertRaises(DataContractError):
                    load_dataset_registry(path)

    def test_api_boundary_rejects_ambiguous_json_before_model_validation(self) -> None:
        for document in (
            b'{"case_id":"first","case_id":"second"}',
            b'{"time_in_state_min":Infinity}',
            b"[]",
        ):
            with self.subTest(document=document), self.assertRaises(StrictJSONError):
                _strict_request_object(document)

        with self.assertRaisesRegex(StrictJSONError, "exceeds"):
            _strict_request_object(b" " * (MAX_ASSESSMENT_JSON_BYTES + 1))

    def test_config_and_file_request_reject_duplicate_keys_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text('{"project":"first","project":"second"}', encoding="utf-8")
            with self.assertRaisesRegex(StrictJSONError, "duplicate JSON object key"):
                load_config(config_path)

            request_path = root / "request.json"
            request_path.write_text('{"case_id":"first","case_id":"second"}', encoding="utf-8")
            with self.assertRaisesRegex(StrictJSONError, "duplicate JSON object key"):
                assess_request(root / "unused-run", request_path)


if __name__ == "__main__":
    unittest.main()
