"""Real offline migration under deployment-owned quiescence and source ownership."""

import contextlib
import unittest
from unittest.mock import Mock, patch

from tests import test_migrate_workspaces_v1 as fixtures
from tools import startup_migration as startup
from tools.migrate_v1_database import MigrationError
from tools.migration_files import inventory_tree
from vibesim_agent.storage.database import Database
from vibesim_agent.storage.ownership import WorkspaceOwnership


class StartupMigrationTests(unittest.TestCase):
    run_migration = fixtures.WorkspaceMigrationTests.run_migration

    def setUp(self):
        fixtures.WorkspaceMigrationTests.setUp(self)
        self.record = self.root / "selection.json"
        self.quiet = False
        self.transitions = []
        self.before = inventory_tree(self.source)

    @contextlib.contextmanager
    def quiesce(self, source):
        self.assertEqual(source, self.source)
        blocked = WorkspaceOwnership(source)
        self.addCleanup(blocked.close)
        with self.assertRaisesRegex(RuntimeError, "owned by another backend"):
            blocked.acquire()
        self.quiet = True
        self.transitions.append("enter")
        try:
            yield
        finally:
            self.transitions.append("exit")
            self.quiet = False

    def prepare(self, **overrides):
        options = {
            "models": self.models,
            "families": self.families,
            "runners": self.runners,
            "quiesce": self.quiesce,
            **overrides,
        }
        return startup.prepare_startup(self.source, self.target, self.record, **options)

    def assert_source_unchanged_and_unlocked(self):
        self.assertEqual(inventory_tree(self.source), self.before)
        owner = WorkspaceOwnership(self.source)
        try:
            owner.acquire()
        finally:
            owner.close()

    def test_first_migration_and_repeated_selection_hold_quiescence_through_body(self):
        migrate = startup.migrate_workspaces
        calls = []

        def real_migration(*args, **kwargs):
            self.assertTrue(self.quiet)
            calls.append(args)
            return migrate(*args, **kwargs)

        with patch.object(startup, "migrate_workspaces", side_effect=real_migration):
            for _ in range(2):
                with self.prepare() as selected:
                    self.assertEqual(selected, self.target)
                    self.assertTrue(self.quiet)
                    self.assertTrue(self.record.is_file())
                    source_owner = WorkspaceOwnership(self.source)
                    with self.assertRaises(RuntimeError):
                        source_owner.acquire()
                    target_owner = WorkspaceOwnership(selected)
                    try:
                        target_owner.acquire()
                    finally:
                        target_owner.close()
                self.assertFalse(self.quiet)
                self.assert_source_unchanged_and_unlocked()
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.transitions, ["enter", "exit", "enter", "exit"])

    def test_current_state_fast_path_does_not_quiesce_or_migrate(self):
        self.run_migration()
        current = self.target
        before = inventory_tree(current)
        quiesce = Mock(side_effect=AssertionError("current state must not quiesce"))
        with (
            patch.object(
                startup,
                "migrate_workspaces",
                side_effect=AssertionError("unexpected migration"),
            ),
            startup.prepare_startup(
                current,
                self.root / "unused",
                self.record,
                models=self.models,
                families=self.families,
                runners=self.runners,
                quiesce=quiesce,
            ) as selected,
        ):
            self.assertEqual(selected, current)
            owner = WorkspaceOwnership(current)
            try:
                owner.acquire()
            finally:
                owner.close()
        quiesce.assert_not_called()
        self.assertFalse(self.record.exists())
        self.assertEqual(inventory_tree(current), before)
        self.assert_source_unchanged_and_unlocked()

    def test_missing_or_failed_quiesce_writes_nothing_and_releases_source_lock(self):
        @contextlib.contextmanager
        def failed(source):
            raise RuntimeError("writers could not stop")
            yield

        for callback, exception in ((None, MigrationError), (failed, RuntimeError)):
            with self.subTest(callback=callback):
                with self.assertRaises(exception), self.prepare(quiesce=callback):
                    self.fail("failed quiescence must not yield")
                self.assertFalse(self.record.exists())
                self.assertFalse(self.target.exists())
                self.assert_source_unchanged_and_unlocked()

    def test_source_lock_conflict_prevents_quiescence_and_writes(self):
        owner = WorkspaceOwnership(self.source)
        owner.acquire()
        try:
            with self.assertRaises(RuntimeError), self.prepare():
                self.fail("conflicting owner must not yield")
            self.assertEqual(self.transitions, [])
            self.assertFalse(self.target.exists())
            self.assertFalse(self.record.exists())
        finally:
            owner.close()
        self.assert_source_unchanged_and_unlocked()

    def test_complete_unpublished_target_is_verified_and_adopted_without_recopy(self):
        self.run_migration()
        before = inventory_tree(self.target)
        with (
            patch.object(
                startup,
                "migrate_workspaces",
                side_effect=AssertionError("must not recopy"),
            ),
            self.prepare() as selected,
        ):
            self.assertEqual(selected, self.target)
            self.assertTrue(self.quiet)
        self.assertTrue(self.record.exists())
        self.assertEqual(inventory_tree(self.target), before)
        self.assert_source_unchanged_and_unlocked()

    def test_partial_target_and_corrupt_record_never_fall_back(self):
        self.target.mkdir()
        sentinel = self.target / "partial"
        sentinel.write_bytes(b"preserve partial evidence")
        with self.assertRaises((MigrationError, OSError)), self.prepare():
            self.fail("partial target must not yield")
        self.assertEqual(sentinel.read_bytes(), b"preserve partial evidence")
        self.assertFalse(self.record.exists())
        self.assert_source_unchanged_and_unlocked()
        self.record.write_bytes(b"corrupt record")
        transitions = list(self.transitions)
        with (
            patch.object(
                startup,
                "migrate_workspaces",
                side_effect=AssertionError("must not recopy"),
            ),
            self.assertRaises(MigrationError),
            self.prepare(),
        ):
            self.fail("corrupt selection must not yield")
        self.assertEqual(self.record.read_bytes(), b"corrupt record")
        self.assertEqual(self.transitions, transitions)
        self.assertEqual(sentinel.read_bytes(), b"preserve partial evidence")
        self.assert_source_unchanged_and_unlocked()

    def test_changed_configured_target_rejects_existing_selection(self):
        with self.prepare():
            pass
        before = self.record.read_bytes(), inventory_tree(self.target)
        original = self.target
        transitions = list(self.transitions)
        self.target = self.root / "different-target"
        with (
            self.assertRaisesRegex(MigrationError, "configured target differs"),
            self.prepare(),
        ):
            self.fail("cannot silently retarget")
        self.assertFalse(self.target.exists())
        self.assertEqual(self.transitions, transitions)
        self.assertEqual((self.record.read_bytes(), inventory_tree(original)), before)
        self.assert_source_unchanged_and_unlocked()

    def test_existing_selection_still_requires_quiescence(self):
        with self.prepare():
            pass
        before = self.record.read_bytes(), inventory_tree(self.target)
        transitions = list(self.transitions)
        with (
            self.assertRaisesRegex(MigrationError, "writer shutdown"),
            self.prepare(quiesce=None),
        ):
            self.fail("selection cannot bypass writer shutdown")
        self.assertEqual(self.transitions, transitions)
        self.assertEqual(
            (self.record.read_bytes(), inventory_tree(self.target)), before
        )
        self.assert_source_unchanged_and_unlocked()

    def test_external_paths_must_match_the_selected_migration(self):
        first = self.root / "external-first"
        second = self.root / "external-second"
        for directory in (first, second):
            (directory / "logs").mkdir(parents=True)

        def paths(directory):
            return {"w_main": {"repo_path": directory, "logs_path": directory / "logs"}}

        with self.prepare(mode="rehearsal", external_paths=paths(first)):
            pass
        before = self.record.read_bytes(), inventory_tree(self.target)
        with (
            self.assertRaisesRegex(MigrationError, "external paths changed"),
            self.prepare(mode="rehearsal", external_paths=paths(second)),
        ):
            self.fail("cannot silently switch external roots")
        self.assertEqual(
            (self.record.read_bytes(), inventory_tree(self.target)), before
        )
        self.assert_source_unchanged_and_unlocked()

    def test_body_failure_preserves_record_and_new_data_for_next_start(self):
        with self.assertRaisesRegex(RuntimeError, "application failed"), self.prepare():
            with Database(self.target / "w_main/workspace.sqlite").connect(
                write=True
            ) as connection:
                connection.execute(
                    "INSERT INTO messages(conversation_id,role,content,ts,metadata_json,turn_id) VALUES ('c','assistant','new data',10,'{}','t')"
                )
            raise RuntimeError("application failed")
        self.assertFalse(self.quiet)
        record = self.record.read_bytes()
        with (
            patch.object(
                startup,
                "migrate_workspaces",
                side_effect=AssertionError("must not rollback"),
            ),
            self.prepare() as selected,
            Database(selected / "w_main/workspace.sqlite").connect() as connection,
        ):
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE content='new data'"
                ).fetchone()[0],
                1,
            )
        self.assertEqual(self.record.read_bytes(), record)
        self.assert_source_unchanged_and_unlocked()
