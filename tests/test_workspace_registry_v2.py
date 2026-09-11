"""Workspace discovery reads the legacy descriptor format without implicit writes."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vibesim_agent.storage.registry import WorkspaceRegistry


class WorkspaceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.root = self.directory / "registry"
        self.registry = WorkspaceRegistry(self.root)

    def write_descriptor(self, workspace_id="w_test", **overrides):
        descriptor = {
            "schema_version": 1,
            "workspace_id": workspace_id,
            "display_name": "Example",
            "naming_state": "manual",
            "state": "active",
            "storage_kind": "managed",
            "repo_path": "repo",
            "logs_path": "repo/logs",
            "created_at": 10,
            "last_accessed_at": 20,
            "base_workspace_id": "w_main",
            "base_revision": "revision",
        }
        descriptor.update(overrides)
        path = self.root / workspace_id / "workspace.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(descriptor), encoding="utf-8")
        return descriptor

    def test_empty_root_and_paths_do_not_create_registry_main_or_database(self):
        self.assertEqual(self.registry.list(), [])
        self.assertEqual(self.registry.registry_path, self.root / "registry.json")
        self.assertEqual(self.registry.workspace_dir("w_test"), self.root / "w_test")
        self.assertEqual(
            self.registry.database_path("w_test"), self.root / "w_test/workspace.sqlite"
        )
        self.assertEqual(self.registry.jobs_dir("w_test"), self.root / "w_test/jobs")
        self.assertEqual(
            self.registry.conversation_runtime_path("w_test", "old:conversation"),
            self.root / "w_test/runtime/old:conversation",
        )
        self.assertEqual(
            self.registry.legacy_runtime_path("w_test", "old:conversation"),
            self.root / "w_test/codex/old:conversation",
        )
        with self.assertRaises(KeyError):
            self.registry.get("w_main")
        self.assertFalse(self.root.exists())

    def test_descriptor_values_paths_and_legacy_naming_default_are_preserved(self):
        descriptor = self.write_descriptor()
        self.assertEqual(self.registry.get("w_test"), descriptor)
        self.assertEqual(self.registry.repo_path("w_test"), self.root / "w_test/repo")
        self.assertEqual(
            self.registry.logs_path("w_test"), self.root / "w_test/repo/logs"
        )
        descriptor.pop("naming_state")
        path = self.registry.descriptor_path("w_test")
        path.write_text(json.dumps(descriptor))
        original = path.read_bytes()
        self.assertEqual(self.registry.get("w_test")["naming_state"], "manual")
        self.assertEqual(path.read_bytes(), original)

    def test_external_main_paths_resolve_relative_and_absolute_values(self):
        main = self.directory / "source"
        self.write_descriptor(
            "w_main",
            storage_kind="external",
            repo_path="../../source",
            logs_path=str(main / "logs"),
        )
        self.assertEqual(self.registry.repo_path("w_main"), main)
        self.assertEqual(self.registry.logs_path("w_main"), main / "logs")
        self.assertFalse(main.exists())

    def test_list_uses_descriptors_main_first_recent_then_id_and_archived_filter(self):
        self.write_descriptor("w_z", last_accessed_at=40)
        self.write_descriptor("w_a", last_accessed_at=40)
        self.write_descriptor("w_old", last_accessed_at=1)
        self.write_descriptor("w_main", last_accessed_at=0)
        self.write_descriptor("w_archived", last_accessed_at=80, state="archived")
        self.registry.registry_path.write_text('{"stale": "index"}')
        self.assertEqual(
            [d["workspace_id"] for d in self.registry.list()],
            ["w_main", "w_a", "w_z", "w_old"],
        )
        self.assertEqual(
            [d["workspace_id"] for d in self.registry.list(include_archived=True)],
            ["w_main", "w_archived", "w_a", "w_z", "w_old"],
        )
        self.assertEqual(self.registry.get("w_archived")["state"], "archived")

    def test_malformed_descriptor_is_skipped_in_list_but_explicit_get_rejects(self):
        valid = self.write_descriptor("w_valid")
        bad = self.write_descriptor("w_bad")
        path = self.registry.descriptor_path("w_bad")
        examples = [
            [],
            {},
            {**bad, "workspace_id": "w_valid"},
            {**bad, "schema_version": 2},
            {**bad, "schema_version": True},
            {**bad, "state": []},
            {**bad, "storage_kind": {}},
            {**bad, "naming_state": []},
            {**bad, "repo_path": None},
            {**bad, "created_at": "yesterday"},
            {**bad, "last_accessed_at": float("nan")},
            {**bad, "last_accessed_at": 10**400},
        ]
        for value in examples:
            with self.subTest(value=value):
                path.write_text(json.dumps(value))
                self.assertEqual(self.registry.list(), [valid])
                with self.assertRaises(ValueError):
                    self.registry.get("w_bad")
        path.write_text("{broken")
        self.assertEqual(self.registry.list(), [valid])

    def test_ids_cannot_traverse_or_escape_runtime_state_paths(self):
        for workspace_id in (
            "main",
            "w_",
            "w_../escape",
            "w_" + "a" * 65,
            None,
            "w_a\n",
        ):
            with self.subTest(workspace_id=workspace_id), self.assertRaises(ValueError):
                self.registry.database_path(workspace_id)
        for conversation_id in (
            "",
            ".",
            "..",
            "a/b",
            "a\\b",
            "x" * 81,
            "a\x00b",
            "a\nb",
            None,
        ):
            with (
                self.subTest(conversation_id=conversation_id),
                self.assertRaises(ValueError),
            ):
                self.registry.conversation_runtime_path("w_test", conversation_id)
        with self.assertRaises(ValueError):
            WorkspaceRegistry(Path("relative"))
        self.assertEqual(self.registry.workspace_dir("w_A-B_c"), self.root / "w_A-B_c")

    def test_internal_state_symlink_escape_rejected_but_explicit_external_repo_allowed(
        self,
    ):
        outside = self.directory / "outside"
        outside.mkdir()
        self.root.mkdir()
        (self.root / "w_escape").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.registry.database_path("w_escape")
        self.write_descriptor(repo_path=str(outside))
        self.assertEqual(self.registry.repo_path("w_test"), outside)
        (self.root / "w_test/runtime").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.registry.conversation_runtime_path("w_test", "conversation")
        (self.root / "w_test/workspace.sqlite").symlink_to(outside / "state.sqlite")
        with self.assertRaises(ValueError):
            self.registry.database_path("w_test")

    def test_reads_leave_descriptor_index_and_sqlite_bytes_unchanged(self):
        self.write_descriptor()
        self.registry.database_path("w_test").write_bytes(b"not opened as sqlite")
        self.registry.registry_path.write_text("old index")
        before = {
            str(p): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in self.root.rglob("*")
            if p.is_file()
        }
        with (
            patch("os.getenv", side_effect=AssertionError("environment read")),
            patch("sqlite3.connect", side_effect=AssertionError("database opened")),
        ):
            self.registry.get("w_test")
            self.registry.list()
            self.registry.repo_path("w_test")
            self.registry.logs_path("w_test")
            self.registry.database_path("w_test")
        self.assertEqual(
            before,
            {
                str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.root.rglob("*")
                if p.is_file()
            },
        )

    def test_module_import_has_no_backend_dependency(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'backend' or name.startswith('backend.'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from vibesim_agent.storage.registry import WorkspaceRegistry
""",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
