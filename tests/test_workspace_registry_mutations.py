import json
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vibesim_agent.storage.database import Database
from vibesim_agent.storage.registry import WorkspaceRegistry


class WorkspaceRegistryMutationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.registry = WorkspaceRegistry(self.root, clock=lambda: 50)
        self.counter = 0

    def descriptor(self, wid="w_test", **overrides):
        return {
            "schema_version": 1,
            "workspace_id": wid,
            "display_name": "Untitled",
            "naming_state": "pending",
            "state": "active",
            "storage_kind": "managed",
            "repo_path": "repo",
            "logs_path": "repo/logs",
            "created_at": 10,
            "last_accessed_at": 10,
            "base_workspace_id": "w_main",
            "base_revision": "base",
            **overrides,
        }

    def stage(self):
        self.counter += 1
        stage = self.root / f".workspace-{self.counter}"
        (stage / "repo").mkdir(parents=True)
        (stage / "repo/AGENTS.md").write_text("instructions")
        Database.create(stage / "workspace.sqlite")
        return stage

    def publish(self, wid="w_test", **overrides):
        return self.registry.publish(self.stage(), self.descriptor(wid, **overrides))

    def index(self):
        return json.loads(self.registry.registry_path.read_text())

    def test_publish_moves_prepared_repo_and_database_then_indexes_legacy_shape(self):
        stage = self.stage()
        descriptor = self.descriptor()
        self.assertEqual(self.registry.list(), [])
        result = self.registry.publish(stage, descriptor)
        self.assertEqual(result, descriptor)
        self.assertEqual(self.registry.get("w_test"), descriptor)
        self.assertEqual(list(stage.iterdir()), [])
        self.assertEqual(
            (self.registry.repo_path("w_test") / "AGENTS.md").read_text(),
            "instructions",
        )
        with Database(self.registry.database_path("w_test")).connect() as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
                0,
            )
        self.assertEqual(
            self.index(),
            {
                "schema_version": 1,
                "workspaces": [
                    {
                        "workspace_id": "w_test",
                        "display_name": "Untitled",
                        "state": "active",
                        "logs_root": "w_test/repo/logs",
                    }
                ],
            },
        )

    def test_existing_empty_directory_and_existing_workspace_are_never_overwritten(
        self,
    ):
        destination = self.root / "w_test"
        destination.mkdir()
        inode = destination.stat().st_ino
        stage = self.stage()
        with self.assertRaises(FileExistsError):
            self.registry.publish(stage, self.descriptor())
        self.assertEqual(destination.stat().st_ino, inode)
        self.assertEqual(list(destination.iterdir()), [])
        self.assertTrue((stage / "workspace.sqlite").is_file())
        destination.rmdir()
        self.registry.publish(stage, self.descriptor())
        before = (destination / "workspace.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual((destination / "workspace.json").read_bytes(), before)

    def test_failed_descriptor_or_index_publish_removes_only_new_target(self):
        self.publish("w_keep")
        index_before = self.registry.registry_path.read_bytes()
        original = self.registry._write_json_atomic
        for failing_name in ("workspace.json", "registry.json"):
            stage = self.stage()

            def fail(path, payload, failing_name=failing_name):
                if path.name == failing_name:
                    raise OSError("write failure")
                return original(path, payload)

            with (
                self.subTest(failing_name=failing_name),
                patch.object(self.registry, "_write_json_atomic", side_effect=fail),
                self.assertRaises(OSError),
            ):
                self.registry.publish(stage, self.descriptor())
            self.assertFalse((self.root / "w_test").exists())
            self.assertTrue((self.root / "w_keep/workspace.json").is_file())
            self.assertEqual(self.registry.registry_path.read_bytes(), index_before)
            self.assertEqual(
                [item["workspace_id"] for item in self.registry.list()], ["w_keep"]
            )

    def test_partial_move_failure_rolls_back_owned_target(self):
        stage = self.stage()
        original = Path.rename
        count = 0

        def fail_second(path, target):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("move failure")
            return original(path, target)

        with patch.object(Path, "rename", fail_second), self.assertRaises(OSError):
            self.registry.publish(stage, self.descriptor())
        self.assertFalse((self.root / "w_test").exists())
        self.assertFalse(self.registry.registry_path.exists())

    def test_update_name_archive_touch_and_generated_cas(self):
        self.publish()
        self.assertTrue(self.registry.apply_generated_name("w_test", " Generated "))
        generated = self.registry.get("w_test")
        self.assertEqual(
            (
                generated["display_name"],
                generated["naming_state"],
                generated["last_accessed_at"],
            ),
            ("Generated", "generated", 50),
        )
        self.assertFalse(self.registry.apply_generated_name("w_test", "later"))
        updated = self.registry.update(
            "w_test", display_name=" Manual ", state="archived", touch=True
        )
        self.assertEqual(
            (updated["display_name"], updated["naming_state"], updated["created_at"]),
            ("Manual", "manual", 10),
        )
        self.assertEqual(self.registry.list(), [])
        self.assertEqual(len(self.registry.list(include_archived=True)), 1)
        self.assertEqual(self.index()["workspaces"][0]["state"], "archived")
        self.registry.update("w_test", state="active")
        self.assertFalse(self.registry.apply_generated_name("w_test", "overwrite"))

    def test_manual_rename_wins_pending_generator(self):
        self.publish()
        self.registry.update("w_test", display_name="Chosen")
        self.assertFalse(self.registry.apply_generated_name("w_test", "Generated"))
        self.assertEqual(self.registry.get("w_test")["display_name"], "Chosen")

    def test_update_index_failure_keeps_descriptor_and_rebuild_recovers(self):
        self.publish()
        original = self.registry._write_json_atomic
        index_before = self.registry.registry_path.read_bytes()

        def fail_index(path, payload):
            if path.name == "registry.json":
                raise OSError("index failure")
            return original(path, payload)

        with (
            patch.object(self.registry, "_write_json_atomic", side_effect=fail_index),
            self.assertRaises(OSError),
        ):
            self.registry.update("w_test", display_name="New name")
        self.assertEqual(self.registry.get("w_test")["display_name"], "New name")
        self.assertEqual(self.registry.registry_path.read_bytes(), index_before)
        self.registry.rebuild_index()
        self.assertEqual(self.index()["workspaces"][0]["display_name"], "New name")

    def test_main_archive_invalid_fields_and_invalid_clock_do_not_write(self):
        main = self.root / "w_main"
        main.mkdir()
        descriptor = self.descriptor(
            "w_main", storage_kind="external", naming_state="manual"
        )
        (main / "workspace.json").write_text(json.dumps(descriptor))
        before = (main / "workspace.json").read_bytes()
        for options in (
            {"state": "archived"},
            {"state": "invalid"},
            {"display_name": " "},
        ):
            with self.assertRaises(ValueError):
                self.registry.update("w_main", **options)
            self.assertEqual((main / "workspace.json").read_bytes(), before)
        self.registry.clock = lambda: float("nan")
        with self.assertRaises(ValueError):
            self.registry.update("w_main", touch=True)
        self.assertEqual((main / "workspace.json").read_bytes(), before)

    def test_invalid_staging_descriptor_and_database_fail_before_reservation(self):
        stage = self.stage()
        for descriptor in (
            self.descriptor(repo_path="../outside"),
            self.descriptor("w_main"),
            self.descriptor(schema_version=2),
            self.descriptor(created_at=float("inf")),
        ):
            with self.assertRaises(ValueError):
                self.registry.publish(stage, descriptor)
        alias = self.root / ".workspace-link"
        alias.symlink_to(stage, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.registry.publish(alias, self.descriptor())
        (stage / "workspace.sqlite").unlink()
        with self.assertRaises(ValueError):
            self.registry.publish(stage, self.descriptor())
        self.assertFalse((self.root / "w_test").exists())
        self.assertFalse(self.registry.registry_path.exists())

    def test_same_instance_reader_waits_until_publication_finishes(self):
        stage = self.stage()
        entered, release, reading = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        original = self.registry._write_json_atomic

        def delayed(path, payload):
            original(path, payload)
            if path.name == "workspace.json":
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("test release missing")

        def read():
            reading.set()
            return self.registry.list()

        with (
            ThreadPoolExecutor(max_workers=2) as pool,
            patch.object(self.registry, "_write_json_atomic", side_effect=delayed),
        ):
            writer = pool.submit(self.registry.publish, stage, self.descriptor())
            try:
                self.assertTrue(entered.wait(3))
                reader = pool.submit(read)
                self.assertTrue(reading.wait(3))
                self.assertFalse(reader.done())
            finally:
                release.set()
            self.assertEqual(writer.result(timeout=3)["workspace_id"], "w_test")
            self.assertEqual(reader.result(timeout=3)[0]["workspace_id"], "w_test")

    def test_fdopen_failure_closes_descriptor_and_removes_temporary_state(self):
        stage = self.stage()
        descriptors = []

        def fail(descriptor, *args, **kwargs):
            descriptors.append(descriptor)
            raise OSError("fdopen failed")

        with (
            patch("vibesim_agent.storage.registry.os.fdopen", side_effect=fail),
            self.assertRaisesRegex(OSError, "fdopen failed"),
        ):
            self.registry.publish(stage, self.descriptor())
        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])
        self.assertFalse((self.root / "w_test").exists())
        self.assertFalse(self.registry.registry_path.exists())
        self.assertEqual(list(self.root.glob(".*.json.*")), [])

    def test_actual_index_replace_failure_preserves_old_index_and_cleans_temp(self):
        self.publish("w_keep")
        before = self.registry.registry_path.read_bytes()
        original = Path.replace

        def fail(path, target):
            if target.name == "registry.json":
                raise OSError("replace failure")
            return original(path, target)

        with (
            patch.object(Path, "replace", fail),
            self.assertRaisesRegex(OSError, "replace failure"),
        ):
            self.publish()
        self.assertEqual(self.registry.registry_path.read_bytes(), before)
        self.assertFalse((self.root / "w_test").exists())
        self.assertEqual(list(self.root.glob(".registry.json.*")), [])


class ExternalWorkspacePublicationTests(unittest.TestCase):
    """External workspaces own a real checkout; the registry only points at it."""

    descriptor = WorkspaceRegistryMutationTests.descriptor

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.registry = WorkspaceRegistry(self.root, clock=lambda: 50)
        self.counter = 0
        self.checkout = Path(self.enterContext(TemporaryDirectory())) / "wt-topic"
        (self.checkout / "logs").mkdir(parents=True)

    def external_stage(self):
        self.counter += 1
        stage = self.root / f".workspace-{self.counter}"
        stage.mkdir()
        Database.create(stage / "workspace.sqlite")
        return stage

    def external(self, **overrides):
        return self.descriptor(
            "w_worktree",
            **{
                "storage_kind": "external",
                "repo_path": str(self.checkout),
                "logs_path": str(self.checkout / "logs"),
                **overrides,
            },
        )

    def test_publishes_without_moving_the_checkout_into_the_registry(self):
        descriptor = self.registry.publish(self.external_stage(), self.external())

        self.assertEqual(descriptor["storage_kind"], "external")
        self.assertEqual(self.registry.repo_path("w_worktree"), self.checkout.resolve())
        self.assertTrue(self.checkout.is_dir())
        # The registry directory holds state only; the tree stays where it is.
        self.assertFalse((self.registry.workspace_dir("w_worktree") / "repo").exists())

    def test_optional_worktree_keys_round_trip_without_a_schema_bump(self):
        descriptor = self.registry.publish(
            self.external_stage(), self.external(worktree_branch="wt", worktree_owned=True)
        )
        stored = self.registry.get("w_worktree")
        self.assertEqual(stored["worktree_branch"], "wt")
        self.assertIs(stored["worktree_owned"], True)
        self.assertEqual(descriptor["schema_version"], 1)
        for invalid in ({"worktree_branch": " "}, {"worktree_owned": "yes"}):
            with self.subTest(**invalid):
                with self.assertRaisesRegex(ValueError, "worktree_"):
                    self.registry.publish(self.external_stage(), self.external(**invalid))

    def test_repository_inside_the_registry_is_refused(self):
        # Deleting a workspace rmtree's its registry directory, so this
        # descriptor would turn a workspace delete into a checkout delete.
        inside = self.registry.workspace_dir("w_worktree")
        (inside / "logs").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "outside the registry"):
            self.registry.publish(
                self.external_stage(),
                self.external(repo_path=str(inside), logs_path=str(inside / "logs")),
            )

    def test_relative_detached_or_missing_paths_are_refused(self):
        cases = {
            "absolute": {"repo_path": "repo", "logs_path": "repo/logs"},
            "logs elsewhere": {"logs_path": str(self.checkout.parent / "logs")},
            "missing": {
                "repo_path": str(self.checkout.parent / "gone"),
                "logs_path": str(self.checkout.parent / "gone/logs"),
            },
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    self.registry.publish(
                        self.external_stage(), self.external(**overrides)
                    )

    def test_staged_repository_is_refused_for_an_external_workspace(self):
        stage = self.external_stage()
        (stage / "repo").mkdir()
        # Moving it in would leave an orphan copy shadowed by the absolute path.
        with self.assertRaisesRegex(ValueError, "must not contain a repository"):
            self.registry.publish(stage, self.external())

    def test_managed_publication_still_requires_its_local_repository(self):
        stage = self.external_stage()
        with self.assertRaisesRegex(ValueError, "requires a repository"):
            self.registry.publish(stage, self.descriptor("w_managed"))
