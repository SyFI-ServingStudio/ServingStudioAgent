"""Atomic identity, session compatibility and turn completion behavior."""

import sqlite3
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import AgentMode, Role
from vibesim_agent.storage.conversations import Conversations
from vibesim_agent.storage.database import Database
from vibesim_agent.storage.sessions import Session, Sessions
from vibesim_agent.storage.turns import Turns


class ConversationStorageTests(unittest.TestCase):
    def test_turn_settings_and_user_message_rollback_together(self):
        before = self.conversations.get("c")
        with self.database.connect(write=True) as connection:
            connection.execute("CREATE TRIGGER reject_user BEFORE INSERT ON messages "
                               "WHEN NEW.role='user' BEGIN SELECT RAISE(ABORT, 'user rejected'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.turns.start("c", "failed", "question", settings={
                "sandbox": "read-only", "autonomous": 1, "agent_mode": "single",
                "interrupted_role": "assistant", "prompt_fingerprint": "new fingerprint",
            })
        self.assertEqual(self.conversations.get("c"), before)
        self.assertIsNone(self.turns.get("c", "failed"))
        self.assertEqual(self.conversations.messages("c"), ())

    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.database = Database.create(root / "workspace.sqlite")
        self.conversations = Conversations(self.database)
        self.turns = Turns(self.database)
        self.runtime = RoleRuntime("provider", "scope", "model", "high", "default")
        self.conversations.create("c", runtimes={role: self.runtime for role in Role},
                                  agent_mode=AgentMode.SINGLE)

    def test_completed_turn_writes_one_answer_and_landing_role_atomically(self):
        self.turns.start("c", "t", "question")
        self.turns.append_event("t", "role_start", {"role": "assistant"})
        self.turns.append_event("t", "role_ready", {"role": "assistant"})
        self.assertTrue(self.turns.finish("c", "t", text="Stopped.", status="complete",
                                         interrupted_role="assistant", metadata={"outcome": "cancelled"}))
        self.assertFalse(self.turns.finish("c", "t", text="duplicate", status="failed"))
        messages = self.conversations.messages("c")
        self.assertEqual([m.content for m in messages], ["question", "Stopped."])
        self.assertEqual([m.turn_id for m in messages], ["t", "t"])
        self.assertEqual(self.conversations.get("c")["interrupted_role"], "assistant")
        with self.assertRaises(ValueError):
            self.turns.append_event("t", "final", {"text": "late"})

    def test_started_scope_lock_keeps_compatible_model_sessions_and_allows_unused_roles(self):
        self.turns.start("c", "t", "question")
        sessions = Sessions(self.database)
        sessions.save("c", Session(Role.ASSISTANT, "provider", "scope", "session"))
        self.conversations.update_runtimes("c", {Role.ASSISTANT: replace(self.runtime, model_id="sibling")})
        self.assertEqual(sessions.compatible("c", {Role.ASSISTANT: "scope"}), {Role.ASSISTANT: "session"})
        other = replace(self.runtime, provider_id="other", session_scope="other")
        with self.assertRaisesRegex(ValueError, "locked"):
            self.conversations.update_runtimes("c", {Role.ASSISTANT: other})
        self.conversations.update_runtimes("c", {Role.IMPLEMENTER: other})
        self.assertEqual(self.conversations.runtimes("c")[Role.IMPLEMENTER], other)

    def test_events_are_ordered_and_never_read_from_another_conversation(self):
        self.turns.start("c", "t", "question")
        for index in range(3):
            self.assertEqual(self.turns.append_event("t", "tool_call", {"text": str(index)}), index)
        self.assertEqual([e["payload"]["text"] for e in self.turns.events("c", "t")], ["0", "1", "2"])
        with self.assertRaises(KeyError):
            self.turns.events("other", "t")
        with self.assertRaises(ValueError):
            self.conversations.append("c", "assistant", "bad", turn_id="missing")

    def test_failed_terminal_insert_rolls_back_turn_status_and_resume_role(self):
        self.turns.start("c", "t", "question")
        with self.database.connect(write=True) as connection:
            connection.execute("""CREATE TRIGGER fail_answer BEFORE INSERT ON messages
                WHEN NEW.role = 'assistant' BEGIN SELECT RAISE(ABORT, 'write failed'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.turns.finish("c", "t", text="answer", status="complete", interrupted_role="assistant")
        self.assertEqual(len(self.conversations.messages("c")), 1)
        self.assertEqual(self.conversations.get("c")["interrupted_role"], "")
        with self.database.connect() as connection:
            self.assertEqual(connection.execute("SELECT status FROM turns WHERE id = 't'").fetchone()[0], "running")

    def test_late_terminal_failure_rolls_back_answer_and_resume_role_together(self):
        self.turns.start("c", "t", "question")
        with self.database.connect(write=True) as connection:
            connection.execute("""CREATE TRIGGER fail_terminal BEFORE UPDATE OF status ON turns
                BEGIN SELECT RAISE(ABORT, 'terminal write failed'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.turns.finish("c", "t", text="answer", status="complete", interrupted_role="assistant")
        self.assertEqual([m.content for m in self.conversations.messages("c")], ["question"])
        self.assertEqual(self.conversations.get("c")["interrupted_role"], "")
        with self.database.connect() as connection:
            self.assertEqual(connection.execute("SELECT status FROM turns WHERE id = 't'").fetchone()[0], "running")

    def test_legacy_nullable_turn_ids_and_message_high_water_remain_stable(self):
        with self.database.connect(write=True) as connection:
            connection.execute("""INSERT INTO messages
                (id, conversation_id, role, content, ts, metadata_json, turn_id)
                VALUES (42, 'c', 'assistant', 'old', 1.25, '{}', NULL)""")
            connection.execute("UPDATE sqlite_sequence SET seq = 100 WHERE name = 'messages'")
        new_id = self.conversations.append("c", "user", "new", timestamp=2.5)
        self.assertEqual(new_id, 101)
        messages = self.conversations.messages("c")
        self.assertEqual([m.id for m in messages], [42, 101])
        self.assertEqual([m.turn_id for m in messages], [None, None])
        self.assertEqual([m.ts for m in messages], [1.25, 2.5])

    def test_failure_preserves_resume_role_until_explicitly_cleared(self):
        with self.database.connect(write=True) as connection:
            connection.execute("UPDATE conversations SET interrupted_role = 'assistant' WHERE id = 'c'")
        self.turns.start("c", "failed", "question")
        self.turns.finish("c", "failed", text="failure", status="failed")
        self.assertEqual(self.conversations.get("c")["interrupted_role"], "assistant")
        self.turns.start("c", "retry", "retry")
        self.turns.finish("c", "retry", text="answer", status="complete", interrupted_role="")
        self.assertEqual(self.conversations.get("c")["interrupted_role"], "")

    def test_turn_lookup_checks_conversation_owner(self):
        self.turns.start("c", "t", "question")
        self.assertEqual(self.turns.get("c", "t")["status"], "running")
        self.assertIsNone(self.turns.get("other", "t"))
        self.assertIsNone(self.turns.get("c", "missing"))

    def test_active_turn_finds_latest_durable_running_turn_only(self):
        self.assertIsNone(self.turns.active("c"))
        self.turns.start("c", "older", "first")
        self.turns.start("c", "newer", "second")
        with self.database.connect(write=True) as connection:
            connection.execute("UPDATE turns SET created_at = 1 WHERE id = 'older'")
            connection.execute("UPDATE turns SET created_at = 2 WHERE id = 'newer'")
        self.assertEqual(Turns(self.database).active("c")["id"], "newer")
        self.assertIsNone(self.turns.active("other"))
        self.turns.finish("c", "newer", text="done", status="complete")
        self.assertEqual(self.turns.active("c")["id"], "older")
        self.turns.finish("c", "older", text="failed", status="failed")
        self.assertIsNone(self.turns.active("c"))

    def test_finish_records_one_done_after_existing_events_and_replays_from_offset(self):
        self.turns.start("c", "t", "question")
        self.turns.append_event("t", "role_start", {"role": "assistant"})
        self.turns.append_event("t", "role_ready", {"role": "assistant"})
        terminal = {"outcome": "answer", "text": "done"}
        self.assertTrue(self.turns.finish("c", "t", text="done", status="complete",
                                         terminal_event=terminal))
        self.assertFalse(self.turns.finish("c", "t", text="duplicate", status="failed",
                                          terminal_event={"outcome": "failed"}))
        self.assertEqual(self.turns.events("c", "t", after_sequence=1),
                         [{"sequence": 2, "kind": "done", "payload": terminal}])
        self.assertEqual(self.turns.events("c", "t", after_sequence=2), [])
        self.assertEqual(len(self.turns.events("c", "t")), 3)
        self.assertEqual([m.content for m in self.conversations.messages("c")], ["question", "done"])
        self.assertEqual(self.turns.get("c", "t")["status"], "complete")
        with self.assertRaises(KeyError):
            self.turns.events("other", "t", after_sequence=1)

    def test_failed_done_insert_rolls_back_message_resume_and_terminal_state(self):
        self.turns.start("c", "t", "question")
        self.turns.append_event("t", "role_start", {"role": "assistant"})
        with self.database.connect(write=True) as connection:
            connection.execute("""CREATE TRIGGER fail_done BEFORE INSERT ON turn_events
                WHEN NEW.kind = 'done' BEGIN SELECT RAISE(ABORT, 'done failed'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.turns.finish("c", "t", text="answer", status="complete",
                              interrupted_role="assistant", terminal_event={})
        self.assertEqual([m.content for m in self.conversations.messages("c")], ["question"])
        self.assertEqual(self.conversations.get("c")["interrupted_role"], "")
        self.assertEqual(self.turns.get("c", "t")["status"], "running")
        self.assertEqual([event["kind"] for event in self.turns.events("c", "t")], ["role_start"])
        with self.database.connect(write=True) as connection:
            connection.execute("DROP TRIGGER fail_done")
        self.assertTrue(self.turns.finish("c", "t", text="answer", status="complete", terminal_event={}))
        self.assertEqual(self.turns.events("c", "t", after_sequence=0),
                         [{"sequence": 1, "kind": "done", "payload": {}}])
