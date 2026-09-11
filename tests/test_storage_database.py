"""Opening state must not migrate it or silently create a replacement database."""

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vibesim_agent.domain.roles import Role
from vibesim_agent.storage.database import Database, SchemaMismatch
from vibesim_agent.storage.sessions import Session, Sessions


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.path = self.root / "workspace.sqlite"

    @staticmethod
    def conversation(database, identity="c"):
        with database.connect(write=True) as connection:
            connection.execute("""INSERT INTO conversations
                (id, title, sandbox, autonomous, agent_mode, created_at, updated_at)
                VALUES (?, 'Title', 'workspace-write', 0, 'single', 1, 1)""", (identity,))

    def test_open_missing_or_legacy_state_does_not_initialize_it(self):
        with self.assertRaises(sqlite3.OperationalError):
            with Database(self.path).connect():
                pass
        self.assertFalse(self.path.exists())
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE legacy (message TEXT)")
            connection.execute("INSERT INTO legacy VALUES ('keep')")
        before = self.path.read_bytes()
        with self.assertRaises(SchemaMismatch):
            with Database(self.path).connect(write=True):
                pass
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            Database.create(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_read_connections_cannot_write_and_failed_transactions_rollback(self):
        database = Database.create(self.path)
        self.conversation(database)
        with self.assertRaises(sqlite3.OperationalError):
            with database.connect() as connection:
                connection.execute("DELETE FROM conversations")
        with self.assertRaisesRegex(RuntimeError, "abort"):
            with database.connect(write=True) as connection:
                connection.execute("DELETE FROM conversations")
                raise RuntimeError("abort")
        with database.connect() as connection:
            self.assertEqual(connection.execute("SELECT title FROM conversations").fetchone()[0], "Title")

    def test_failed_initialization_never_publishes_partial_database_and_can_retry(self):
        with patch.object(Path, "read_text", return_value="CREATE TABLE unfinished(id); invalid SQL;"):
            with self.assertRaises(sqlite3.OperationalError):
                Database.create(self.path)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.root.iterdir()), [])
        database = Database.create(self.path)
        self.conversation(database)
        with database.connect() as connection:
            self.assertEqual(connection.execute("SELECT id FROM conversations").fetchone()[0], "c")

    def test_sessions_isolate_conversations_roles_and_compatibility_scope(self):
        database = Database.create(self.path)
        self.conversation(database)
        self.conversation(database, "other")
        sessions = Sessions(database)
        sessions.save("c", Session(Role.ASSISTANT, "provider", "adapter:provider:v1", "session-a"))
        sessions.save("other", Session(Role.ASSISTANT, "provider", "adapter:provider:v1", "session-b"))
        self.assertEqual(sessions.compatible("c", {Role.ASSISTANT: "adapter:provider:v1"}),
                         {Role.ASSISTANT: "session-a"})
        self.assertEqual(sessions.compatible("c", {Role.ASSISTANT: "adapter:other:v1"}), {})
        self.assertEqual(sessions.compatible("c", {Role.IMPLEMENTER: "adapter:provider:v1"}), {})
        with database.connect(write=True) as connection:
            connection.execute("UPDATE conversations SET prompt_fingerprint = 'new' WHERE id = 'c'")
        self.assertEqual(sessions.compatible("c", {Role.ASSISTANT: "adapter:provider:v1"}),
                         {Role.ASSISTANT: "session-a"})
        with self.assertRaises(sqlite3.IntegrityError):
            sessions.save("missing", Session(Role.ASSISTANT, "provider", "scope", "bad"))
