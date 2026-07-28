from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from finops_r.artifacts import (
    ArtifactVerificationError,
    artifact_manifest,
    atomic_joblib,
    atomic_json,
    cleanup_empty_joblib_staging,
    publish_verified_tree,
    verify_artifact_manifest,
)
from finops_r.json_safety import strict_json_load


class ArtifactTests(unittest.TestCase):
    def _verified_tree(self, root: Path) -> None:
        (root / "payload.txt").write_text("payload", encoding="utf-8")
        atomic_json(root / "artifact_manifest.json", artifact_manifest(root))
        self.assertEqual(verify_artifact_manifest(root)["status"], "PASS")

    def test_exact_inventory_rejects_untracked_and_tampered_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._verified_tree(root)
            (root / "untracked.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactVerificationError, "inventory mismatch"):
                verify_artifact_manifest(root)
            (root / "untracked.txt").unlink()
            (root / "payload.txt").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(
                ArtifactVerificationError, "size mismatch|checksum mismatch"
            ):
                verify_artifact_manifest(root)

    def test_manifest_rejects_unsafe_paths_and_invalid_record_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.txt").write_text("payload", encoding="utf-8")
            records = artifact_manifest(root)
            records[0]["path"] = "../payload.txt"
            atomic_json(root / "artifact_manifest.json", records)
            with self.assertRaisesRegex(ArtifactVerificationError, "unsafe"):
                verify_artifact_manifest(root)
            atomic_json(
                root / "artifact_manifest.json",
                [{"path": "payload.txt", "bytes": 7, "sha256": "0" * 64, "extra": True}],
            )
            with self.assertRaisesRegex(ArtifactVerificationError, "schema"):
                verify_artifact_manifest(root)

    def test_symlink_and_hidden_content_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.txt").write_text("payload", encoding="utf-8")
            (root / ".hidden").write_text("residue", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactVerificationError, "hidden"):
                artifact_manifest(root)
            (root / ".hidden").unlink()
            try:
                (root / "link.txt").symlink_to(root / "payload.txt")
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(ArtifactVerificationError, "symlink"):
                artifact_manifest(root)

    def test_special_filesystem_objects_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            special = root / "special-object"
            special.write_text("placeholder", encoding="utf-8")
            real_stat = Path.stat

            def special_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
                result = real_stat(path, *args, **kwargs)
                if path == special:
                    return os.stat_result((stat.S_IFIFO | 0o600, *result[1:]))
                return result

            with (
                patch.object(Path, "stat", special_stat),
                self.assertRaisesRegex(ArtifactVerificationError, "special filesystem object"),
            ):
                artifact_manifest(root)

    def test_hardlinked_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            linked = root / "linked.txt"
            source.write_text("shared inode", encoding="utf-8")
            try:
                os.link(source, linked)
            except OSError:
                self.skipTest("hardlinks unavailable")
            with self.assertRaisesRegex(ArtifactVerificationError, "hardlink"):
                artifact_manifest(root)

    def test_manifest_rejects_cross_platform_unsafe_names_and_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "AUX.txt").write_text("unsafe", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactVerificationError, "portable"):
                artifact_manifest(root)
            (root / "AUX.txt").unlink()
            (root / "Data.txt").write_text("one", encoding="utf-8")
            records = artifact_manifest(root)
            records.append({**records[0], "path": "data.txt"})
            atomic_json(root / "artifact_manifest.json", records)
            with self.assertRaisesRegex(ArtifactVerificationError, "collision"):
                verify_artifact_manifest(root)

    def test_atomic_json_failure_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "value.json"
            with self.assertRaises(ValueError):
                atomic_json(target, {"invalid": float("nan")})
            self.assertFalse(target.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_atomic_json_retries_transient_windows_sharing_violation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "value.json"
            real_replace = os.replace
            attempts = 0

            def replace_after_transient_failure(source: Path, destination: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError("simulated Windows sharing violation")
                real_replace(source, destination)

            with (
                patch("finops_r.artifacts._IS_WINDOWS", True),
                patch(
                    "finops_r.artifacts.os.replace",
                    side_effect=replace_after_transient_failure,
                ),
                patch("finops_r.artifacts.time.sleep") as sleep,
            ):
                atomic_json(target, {"status": "PASS"})

            self.assertEqual(attempts, 2)
            sleep.assert_called_once()
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["status"], "PASS")

    def test_strict_json_path_read_retries_transient_windows_sharing_violation(self) -> None:
        transient = PermissionError(13, "simulated Windows sharing violation")
        transient.winerror = 32  # type: ignore[attr-defined]
        with (
            patch("finops_r.json_safety._IS_WINDOWS", True),
            patch.object(
                Path,
                "read_text",
                side_effect=[transient, '{"status":"PASS"}'],
            ) as read_text,
            patch("finops_r.json_safety.time.sleep") as sleep,
        ):
            self.assertEqual(strict_json_load(Path("pointer.json")), {"status": "PASS"})

        self.assertEqual(read_text.call_count, 2)
        sleep.assert_called_once()

    def test_strict_json_path_read_retries_windows_eacces_without_winerror(self) -> None:
        transient = PermissionError(13, "simulated translated Windows access denial")
        self.assertIsNone(getattr(transient, "winerror", None))
        with (
            patch("finops_r.json_safety._IS_WINDOWS", True),
            patch.object(
                Path,
                "read_text",
                side_effect=[transient, '{"status":"PASS"}'],
            ) as read_text,
            patch("finops_r.json_safety.time.sleep") as sleep,
        ):
            self.assertEqual(strict_json_load(Path("pointer.json")), {"status": "PASS"})

        self.assertEqual(read_text.call_count, 2)
        sleep.assert_called_once()

    def test_strict_json_path_read_does_not_retry_unrelated_permission_error(self) -> None:
        denied = PermissionError(1, "simulated unrelated permission denial")
        denied.winerror = 1314  # type: ignore[attr-defined]
        with (
            patch("finops_r.json_safety._IS_WINDOWS", True),
            patch.object(Path, "read_text", side_effect=denied) as read_text,
            patch("finops_r.json_safety.time.sleep") as sleep,
            self.assertRaises(PermissionError),
        ):
            strict_json_load(Path("pointer.json"))

        read_text.assert_called_once()
        sleep.assert_not_called()

    def test_strict_json_path_read_eacces_retry_is_bounded_by_deadline(self) -> None:
        denied = PermissionError(13, "simulated persistent access denial")
        denied.winerror = 1314  # type: ignore[attr-defined]
        with (
            patch("finops_r.json_safety._IS_WINDOWS", True),
            patch("finops_r.json_safety._WINDOWS_READ_RETRY_SECONDS", 0.0),
            patch.object(Path, "read_text", side_effect=denied) as read_text,
            patch("finops_r.json_safety.time.sleep") as sleep,
            self.assertRaises(PermissionError),
        ):
            strict_json_load(Path("pointer.json"))

        read_text.assert_called_once()
        sleep.assert_not_called()

    def test_joblib_staging_is_removed_and_cleanup_is_owner_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / ".owner.staging.token"
            (run / "models").mkdir(parents=True)
            atomic_joblib(run / "models" / "model.joblib", {"value": 1}, staging_root=root)
            self.assertTrue((run / "models" / "model.joblib").is_file())
            self.assertEqual(
                [path for path in root.iterdir() if ".joblib.staging." in path.name], []
            )
            empty = root / ".owner.empty.joblib.staging.token"
            empty.mkdir()
            other = root / ".other.empty.joblib.staging.token"
            other.mkdir()
            self.assertEqual(cleanup_empty_joblib_staging(root, "owner"), [empty.name])
            self.assertTrue(other.exists())

    def test_verified_publication_reverifies_and_removes_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            destination = root / "release"
            staging.mkdir()
            self._verified_tree(staging)
            publish_verified_tree(staging, destination)
            self.assertFalse(staging.exists())
            self.assertEqual(verify_artifact_manifest(destination)["status"], "PASS")
            self.assertEqual(list(root.glob(".release.publishing.*")), [])

    def test_failed_publication_never_exposes_destination_or_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            destination = root / "release"
            staging.mkdir()
            self._verified_tree(staging)
            with patch("finops_r.artifacts.shutil.copytree", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    publish_verified_tree(staging, destination)
            self.assertFalse(destination.exists())
            self.assertTrue(staging.exists())
            self.assertEqual(list(root.glob(".release.publishing.*")), [])

    def test_staging_cleanup_failure_cannot_leave_a_visible_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            destination = root / "release"
            staging.mkdir()
            self._verified_tree(staging)
            real_rmtree = shutil.rmtree

            def fail_only_for_staging(path: str | Path, *args: object, **kwargs: object) -> None:
                if Path(path) == staging:
                    raise OSError("injected staging cleanup failure")
                real_rmtree(path, *args, **kwargs)

            with patch("finops_r.artifacts.shutil.rmtree", side_effect=fail_only_for_staging):
                with self.assertRaisesRegex(OSError, "injected staging cleanup failure"):
                    publish_verified_tree(staging, destination)

            self.assertFalse(destination.exists())
            self.assertTrue(staging.exists())
            self.assertEqual(list(root.glob(".release.publishing.*")), [])

    def test_reader_observes_absent_or_fully_verified_published_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            destination = root / "release"
            staging.mkdir()
            for index in range(200):
                (staging / f"payload-{index:03d}.txt").write_text("x" * 4096, encoding="utf-8")
            atomic_json(staging / "artifact_manifest.json", artifact_manifest(staging))
            stop = threading.Event()
            failures: list[BaseException] = []

            def read_until_done() -> None:
                while not stop.is_set():
                    if destination.exists():
                        try:
                            verify_artifact_manifest(destination)
                        except BaseException as exc:
                            failures.append(exc)
                            stop.set()

            reader = threading.Thread(target=read_until_done)
            reader.start()
            publish_verified_tree(staging, destination)
            stop.set()
            reader.join(timeout=5)
            self.assertEqual(failures, [])
            self.assertEqual(verify_artifact_manifest(destination)["status"], "PASS")

    def test_manifest_json_is_parseable_after_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            atomic_json(path, {"status": "PASS"})
            self.assertNotIn(b"\r\n", path.read_bytes())
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "PASS")

    def test_concurrent_atomic_json_readers_never_observe_partial_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pointer.json"
            atomic_json(path, {"writer": -1, "sequence": -1, "payload": "seed"})
            stop = threading.Event()
            failures: list[BaseException] = []

            def read_until_done() -> None:
                while not stop.is_set():
                    try:
                        value = strict_json_load(path)
                        if set(value) != {"writer", "sequence", "payload"}:
                            raise AssertionError(f"partial schema observed: {value}")
                    except BaseException as exc:  # captured and asserted on the main thread
                        failures.append(exc)
                        stop.set()

            def write_many(writer: int) -> None:
                for sequence in range(75):
                    atomic_json(
                        path,
                        {"writer": writer, "sequence": sequence, "payload": "x" * 2048},
                    )

            reader = threading.Thread(target=read_until_done)
            reader.start()
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(write_many, range(4)))
            finally:
                stop.set()
                reader.join(timeout=5)

            self.assertFalse(reader.is_alive())
            self.assertEqual(failures, [])
            final = strict_json_load(path)
            self.assertIn(final["writer"], range(4))
            self.assertEqual(final["sequence"], 74)
            self.assertEqual(
                [item for item in path.parent.iterdir() if item.name.startswith(".pointer.json.")],
                [],
            )


if __name__ == "__main__":
    unittest.main()
