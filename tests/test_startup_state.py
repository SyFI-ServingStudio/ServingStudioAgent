"""Startup format detection reads isolated copies, including committed WAL changes."""

import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.legacy_workspaces import create_legacy_workspace
from tools.migrate_v1_database import MigrationError
from tools.migration_files import inventory_tree
from tools.startup_state import inspect_state
from vibesim_agent.storage.database import APPLICATION_ID, SCHEMA_VERSION, Database


class StartupStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = Path(self.enterContext(TemporaryDirectory()))
        self.root = self.temporary / "state"
        self.main = self.temporary / "main"
        self.main.mkdir()
        self.database = create_legacy_workspace(self.root, self.main)

    def make_current(self, path):
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(str(path) + suffix).unlink(missing_ok=True)
        Database.create(path)

    def inspect_unchanged(self, expected=None):
        before = inventory_tree(self.root)
        children = set(self.temporary.iterdir())
        try:
            if expected is None:
                with self.assertRaises(MigrationError):
                    inspect_state(self.root)
            else:
                self.assertEqual(inspect_state(self.root), expected)
        finally:
            self.assertEqual(inventory_tree(self.root), before)
            self.assertEqual(set(self.temporary.iterdir()), children)

    def add_archived(self):
        database = create_legacy_workspace(
            self.root, self.main, workspace_id="w_archived", name="Archived"
        )
        path = self.root / "w_archived/workspace.json"
        descriptor = json.loads(path.read_text())
        descriptor["state"] = "archived"
        path.write_text(json.dumps(descriptor))
        return database

    def test_real_legacy_and_current_detection_preserves_all_bytes_and_mtimes(self):
        (self.root / "unrelated").write_bytes(b"opaque state\x00")
        self.inspect_unchanged("legacy-v8")
        self.inspect_unchanged("legacy-v8")
        self.make_current(self.database)
        self.inspect_unchanged("current")
        self.inspect_unchanged("current")

    def test_archived_workspaces_are_classified_and_mixed_formats_refuse(self):
        archived = self.add_archived()
        self.inspect_unchanged("legacy-v8")
        self.make_current(archived)
        self.inspect_unchanged()
        self.make_current(self.database)
        self.inspect_unchanged("current")
        with closing(sqlite3.connect(archived)) as connection:
            connection.execute("PRAGMA user_version=999")
            connection.commit()
        self.inspect_unchanged()

    def test_unknown_legacy_version_and_corruption_refuse_without_writes(self):
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "INSERT INTO schema_migrations(version,applied_at) VALUES (99,1)"
            )
            connection.commit()
        self.inspect_unchanged()
        self.database.write_bytes(b"not a SQLite database")
        self.inspect_unchanged()

    def test_current_version_stamps_do_not_disguise_legacy_schema(self):
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.commit()
        self.inspect_unchanged()

    def test_current_logical_turn_links_reject_dangling_and_cross_conversation(self):
        for turn_id in ("missing", "other-turn"):
            with self.subTest(turn_id=turn_id):
                self.make_current(self.database)
                with closing(sqlite3.connect(self.database)) as connection:
                    connection.execute("PRAGMA foreign_keys=ON")
                    connection.executemany(
                        "INSERT INTO conversations(id,title,sandbox,autonomous,agent_mode,created_at,updated_at) VALUES (?,'Fixture','read-only',0,'single',1,1)",
                        [("owner",), ("other",)],
                    )
                    connection.execute(
                        "INSERT INTO turns VALUES ('other-turn','other','complete',1,1)"
                    )
                    connection.execute(
                        "INSERT INTO messages(conversation_id,role,content,ts,metadata_json,turn_id) VALUES ('owner','assistant','saved',1,'{}',?)",
                        (turn_id,),
                    )
                    connection.commit()
                    self.assertEqual(
                        connection.execute("PRAGMA foreign_key_check").fetchall(), []
                    )
                self.inspect_unchanged()

    def test_current_orphaned_experiment_job_remains_supported(self):
        self.make_current(self.database)
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "INSERT INTO experiments VALUES ('e','logs/experiment','ready','agent','deleted-job',1,2)"
            )
            connection.commit()
        self.inspect_unchanged("current")

    def test_uncheckpointed_wal_schema_change_is_seen_without_touching_wal_or_shm(self):
        self.make_current(self.database)
        with closing(sqlite3.connect(self.database)) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            base = self.database.read_bytes()
            writer.execute(
                "ALTER TABLE messages ADD COLUMN future TEXT GENERATED ALWAYS AS (content) VIRTUAL"
            )
            writer.commit()
            self.assertEqual(self.database.read_bytes(), base)
            self.assertGreater(Path(str(self.database) + "-wal").stat().st_size, 0)
            self.assertTrue(Path(str(self.database) + "-shm").is_file())
            self.inspect_unchanged()

    def test_legacy_wal_migration_version_is_seen(self):
        with closing(sqlite3.connect(self.database)) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            base = self.database.read_bytes()
            self.inspect_unchanged("legacy-v8")
            writer.execute(
                "INSERT INTO schema_migrations(version,applied_at) VALUES (99,1)"
            )
            writer.commit()
            self.assertEqual(self.database.read_bytes(), base)
            self.inspect_unchanged()

    def test_missing_and_symlink_roots_do_not_create_or_modify_state(self):
        missing = self.temporary / "missing"
        with self.assertRaises(MigrationError):
            inspect_state(missing)
        self.assertFalse(missing.exists())
        before = inventory_tree(self.root)
        alias = self.temporary / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(MigrationError):
            inspect_state(alias)
        self.assertTrue(alias.is_symlink())
        self.assertEqual(inventory_tree(self.root), before)
        dangling = self.temporary / "dangling"
        dangling.symlink_to(missing, target_is_directory=True)
        with self.assertRaises(MigrationError):
            inspect_state(dangling)
        self.assertTrue(dangling.is_symlink())
        self.assertFalse(missing.exists())

    def test_missing_database_and_database_symlink_refuse(self):
        original = self.temporary / "original.sqlite"
        self.database.rename(original)
        self.inspect_unchanged()
        self.database.symlink_to(original)
        before = original.read_bytes(), original.stat().st_mtime_ns
        self.inspect_unchanged()
        self.assertEqual((original.read_bytes(), original.stat().st_mtime_ns), before)
