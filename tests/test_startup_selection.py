"""Durable startup selection uses verified migration identity, not stale content hashes."""

import json
import os
import shutil
import stat
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_migrate_workspaces_v1 as fixtures
from tools import startup_selection as selection
from tools.migrate_v1_database import MigrationError, ProviderIdentity
from tools.migration_files import inventory_tree
from vibesim_agent.storage.database import Database


class StartupSelectionTests(unittest.TestCase):
    run_migration = fixtures.WorkspaceMigrationTests.run_migration

    def setUp(self):
        fixtures.WorkspaceMigrationTests.setUp(self)
        self.run_migration()
        self.record = self.root / "selected.json"
        self.source_before = inventory_tree(self.source)

    def options(self, **overrides):
        return {
            "models": self.models,
            "families": self.families,
            "runners": self.runners,
            **overrides,
        }

    def publish(self, **overrides):
        return selection.publish_selection(
            self.record, self.source, self.target, **self.options(**overrides)
        )

    def load(self, **overrides):
        return selection.load_selection(
            self.record, self.source, **self.options(**overrides)
        )

    def assert_source_unchanged(self):
        self.assertEqual(inventory_tree(self.source), self.source_before)

    def test_real_migration_publication_allows_later_valid_writes_without_fallback(
        self,
    ):
        self.assertIsNone(self.load())
        self.assertFalse(self.record.exists())
        self.assertEqual(self.publish(), self.target)
        self.assertEqual(stat.S_IMODE(self.record.stat().st_mode), 0o600)
        record = self.record.read_bytes()
        with Database(self.target / "w_main/workspace.sqlite").connect(
            write=True
        ) as connection:
            connection.execute(
                "INSERT INTO messages(conversation_id,role,content,ts,metadata_json,turn_id) VALUES ('c','assistant','new-version message',9,'{}','t')"
            )
        self.assertEqual(self.load(), self.target)
        self.assertEqual(self.record.read_bytes(), record)
        self.assert_source_unchanged()

    def test_changed_mapping_runner_or_mode_is_never_accepted(self):
        self.publish()
        before = self.record.read_bytes()
        for options in (
            {
                "models": {
                    "old-model": ProviderIdentity("model_provider", "other-scope")
                }
            },
            {
                "families": {
                    "old-family": ProviderIdentity("session_provider", "other-scope")
                }
            },
            {"runners": {"old-family": "claude"}},
            {"mode": "rehearsal"},
        ):
            with self.subTest(options=options):
                with self.assertRaises(MigrationError):
                    self.load(**options)
                self.assertEqual(self.record.read_bytes(), before)
        self.assert_source_unchanged()

    def test_initial_publication_checks_report_mapping_and_runner(self):
        for options in (
            {"models": {"old-model": ProviderIdentity("model_provider", "different")}},
            {
                "families": {
                    "old-family": ProviderIdentity("session_provider", "different")
                }
            },
            {"runners": {"old-family": "claude"}},
        ):
            with self.subTest(options=options):
                with self.assertRaises(MigrationError):
                    self.publish(**options)
                self.assertFalse(self.record.exists())
        self.assert_source_unchanged()

    def test_source_change_since_migration_blocks_initial_publication(self):
        changed = self.source / "unknown-root-file"
        changed.write_bytes(b"legacy writer changed state")
        before = inventory_tree(self.source)
        with self.assertRaises(MigrationError):
            self.publish()
        self.assertFalse(self.record.exists())
        self.assertEqual(inventory_tree(self.source), before)

    def test_failed_or_mismatched_migration_report_cannot_be_published(self):
        manifest = self.target / ".migration-v1/manifest.json"
        original = json.loads(manifest.read_text())
        for field, value in (
            ("verified", False),
            ("source_quiesced_by_caller", False),
            ("databases", {}),
            ("target", str(self.source)),
            ("files", []),
            ("descriptors", {**original["descriptors"], "w_main": []}),
            ("homes", [{"provider_id": [], "runner": "codex"}]),
        ):
            with self.subTest(field=field):
                manifest.write_text(json.dumps({**original, field: value}))
                with self.assertRaises(MigrationError):
                    self.publish()
                self.assertFalse(self.record.exists())
        self.assert_source_unchanged()

    def test_target_content_changed_before_selection_refuses_publication(self):
        (self.target / "unknown-root-file").write_bytes(b"changed after migration")
        with self.assertRaisesRegex(MigrationError, "changed before publication"):
            self.publish()
        self.assertFalse(self.record.exists())
        self.assert_source_unchanged()

    def test_record_manifest_and_directory_identity_changes_refuse_load(self):
        self.publish()
        record = self.record.read_bytes()
        self.record.write_bytes(b'{"format":')
        with self.assertRaises(MigrationError):
            self.load()
        self.record.write_bytes(record)
        manifest = self.target / ".migration-v1/manifest.json"
        original = manifest.read_bytes()
        manifest.write_bytes(original + b"\n")
        with self.assertRaisesRegex(MigrationError, "report changed"):
            self.load()
        manifest.write_bytes(original)
        moved = self.root / "original-target"
        self.target.rename(moved)
        shutil.copytree(moved, self.target, symlinks=True)
        with self.assertRaisesRegex(
            MigrationError, "identity or configuration changed"
        ):
            self.load()
        self.assertEqual(self.record.read_bytes(), record)
        self.assert_source_unchanged()

    def test_load_does_not_fall_back_when_selected_database_becomes_invalid(self):
        self.publish()
        (self.target / "w_managed/workspace.sqlite").write_bytes(b"corrupt")
        with self.assertRaises(MigrationError):
            self.load()
        self.assert_source_unchanged()

    def test_record_locations_and_symlinks_are_rejected(self):
        for path in (
            self.source / "selection.json",
            self.target / "selection.json",
            self.main / "selection.json",
            self.main / "logs/selection.json",
        ):
            with self.subTest(path=path):
                with self.assertRaises(MigrationError):
                    selection.publish_selection(
                        path, self.source, self.target, **self.options()
                    )
                self.assertFalse(path.exists())
        self.record.symlink_to(self.root / "absent")
        for operation in (self.publish, self.load):
            with self.assertRaises(MigrationError):
                operation()
        self.assertTrue(self.record.is_symlink())
        self.record.unlink()
        target_alias = self.root / "target-alias"
        target_alias.symlink_to(self.target, target_is_directory=True)
        with self.assertRaises(MigrationError):
            selection.publish_selection(
                self.record, self.source, target_alias, **self.options()
            )
        self.assert_source_unchanged()

    def test_existing_selection_is_never_overwritten(self):
        self.publish()
        original = self.record.read_bytes()
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual(self.record.read_bytes(), original)
        self.assertEqual(self.load(), self.target)
        self.assertEqual(list(self.root.glob(".agent-startup-selection-*")), [])
        self.assert_source_unchanged()

    def test_fsync_failure_before_or_after_publication_has_recoverable_visibility(self):
        fsync = os.fsync
        with (
            patch.object(
                selection.os, "fsync", side_effect=OSError("file fsync failure")
            ),
            self.assertRaises(OSError),
        ):
            self.publish()
        self.assertFalse(self.record.exists())
        self.assertEqual(list(self.root.glob(".agent-startup-selection-*")), [])

        def fail_directory(descriptor):
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError("directory fsync failure")
            return fsync(descriptor)

        with (
            patch.object(selection.os, "fsync", side_effect=fail_directory),
            self.assertRaisesRegex(OSError, "directory fsync"),
        ):
            self.publish()
        self.assertTrue(self.record.is_file())
        original = self.record.read_bytes()
        self.assertEqual(self.load(), self.target)
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual(self.record.read_bytes(), original)
        self.assertEqual(list(self.root.glob(".agent-startup-selection-*")), [])
        self.assert_source_unchanged()

    def test_fdopen_failures_close_raw_descriptors_and_clean_temporary_files(self):
        original_fdopen = os.fdopen
        failed = []

        def fail_publication(descriptor, *args, **kwargs):
            if Path(os.readlink(f"/proc/self/fd/{descriptor}")).name.startswith(
                ".agent-startup-selection-"
            ):
                failed.append(descriptor)
                raise OSError("injected publication fdopen")
            return original_fdopen(descriptor, *args, **kwargs)

        with (
            patch.object(selection.os, "fdopen", side_effect=fail_publication),
            self.assertRaisesRegex(OSError, "publication fdopen"),
        ):
            self.publish()
        self.assertEqual(len(failed), 1)
        with self.assertRaises(OSError):
            os.fstat(failed[0])
        self.assertFalse(self.record.exists())
        self.assertEqual(list(self.root.glob(".agent-startup-selection-*")), [])
        self.publish()
        original = self.record.read_bytes()

        def fail_read(descriptor, *args, **kwargs):
            failed.append(descriptor)
            raise OSError("injected read fdopen")

        with (
            patch.object(selection.os, "fdopen", side_effect=fail_read),
            self.assertRaisesRegex(OSError, "read fdopen"),
        ):
            self.load()
        self.assertEqual(len(failed), 2)
        with self.assertRaises(OSError):
            os.fstat(failed[-1])
        self.assertEqual(self.record.read_bytes(), original)
        self.assert_source_unchanged()
