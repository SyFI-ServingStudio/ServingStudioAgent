"""Conversation and message persistence with explicit model selections."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from typing import Any

from ..domain.conversations import Message, RoleRuntime
from ..domain.roles import AgentMode, Role, Sandbox
from .database import Database


def read_message(row: sqlite3.Row) -> Message:
    # Legacy history tolerated unusable metadata; keep that tolerance on reads
    # without changing the archived bytes or losing valid key/value sequences.
    try:
        metadata = dict(json.loads(row["metadata_json"]))
    except (TypeError, ValueError):
        metadata = {}
    return Message(row["id"], row["role"], row["content"], row["ts"], row["turn_id"],
                   metadata)


def insert_message(connection: sqlite3.Connection, conversation_id: str, role: str,
                   content: str, *, timestamp: float, turn_id: str | None,
                   metadata: Mapping[str, Any]) -> int:
    cursor = connection.execute(
        """INSERT INTO messages(conversation_id, role, content, ts, metadata_json, turn_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (conversation_id, role, content, timestamp,
         json.dumps(dict(metadata), ensure_ascii=False), turn_id),
    )
    connection.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                       (timestamp, conversation_id))
    assert cursor.lastrowid is not None
    return cursor.lastrowid


class Conversations:
    def __init__(self, database: Database):
        self.database = database

    def list(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT id, title, naming_state, updated_at FROM conversations "
                "ORDER BY updated_at DESC, id"
            ).fetchall()
        return [dict(row) for row in rows]

    def page(self, conversation_id: str, *, limit: int, before: int | None = None):
        if limit < 1 or (before is not None and before < 0):
            raise ValueError("invalid message page")
        with self.database.connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                                       (conversation_id,)).fetchone()[0]
            end = total if before is None else min(before, total)
            start = max(0, end - limit)
            rows = connection.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY id LIMIT ? OFFSET ?",
                                      (conversation_id, end - start, start)).fetchall()
        return tuple(read_message(row) for row in rows), {
            "start_index": start, "end_index": end, "total_messages": total, "has_more": start > 0,
        }

    def create(self, conversation_id: str, *, runtimes: Mapping[Role, RoleRuntime],
               agent_mode: AgentMode = AgentMode.ORCHESTRATED,
               sandbox: Sandbox = Sandbox.WORKSPACE_WRITE, autonomous: bool = False,
               title: str = "New conversation", naming_state: str = "manual",
               prompt_fingerprint: str | None = None, peer_workspace: str | None = None) -> None:
        if not conversation_id or not set(agent_mode.roles).issubset(runtimes):
            raise ValueError("conversation identity and all active role selections are required")
        if naming_state not in {"pending", "generated", "manual"}:
            raise ValueError("invalid naming state")
        now = time.time()
        with self.database.connect(write=True) as connection:
            connection.execute("""INSERT INTO conversations
                (id, title, naming_state, sandbox, autonomous, agent_mode, prompt_fingerprint,
                 peer_workspace, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (conversation_id, title, naming_state, sandbox.value, int(autonomous),
                 agent_mode.value, prompt_fingerprint, peer_workspace, now, now))
            for role, runtime in runtimes.items():
                self._write_runtime(connection, conversation_id, role, runtime)

    @staticmethod
    def _write_runtime(connection, conversation_id, role, runtime):
        connection.execute("""INSERT INTO role_settings
            (conversation_id, role, provider_id, session_scope, model_id, effort, service_tier)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id, role) DO UPDATE SET
              provider_id = excluded.provider_id, session_scope = excluded.session_scope,
              model_id = excluded.model_id, effort = excluded.effort, service_tier = excluded.service_tier""",
            (conversation_id, role.value, runtime.provider_id, runtime.session_scope,
             runtime.model_id, runtime.effort, runtime.service_tier))

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM conversations WHERE id = ?",
                                     (conversation_id,)).fetchone()
            return dict(row) if row is not None else None

    def runtimes(self, conversation_id: str) -> dict[Role, RoleRuntime]:
        with self.database.connect() as connection:
            rows = connection.execute("SELECT * FROM role_settings WHERE conversation_id = ?",
                                      (conversation_id,)).fetchall()
        return {Role(row["role"]): RoleRuntime(row["provider_id"], row["session_scope"],
                row["model_id"], row["effort"], row["service_tier"]) for row in rows}

    def update_runtimes(self, conversation_id: str, runtimes: Mapping[Role, RoleRuntime]) -> None:
        with self.database.connect(write=True) as connection:
            conversation = connection.execute("SELECT agent_mode FROM conversations WHERE id = ?",
                                              (conversation_id,)).fetchone()
            if conversation is None:
                raise KeyError(conversation_id)
            started = connection.execute("""SELECT
                EXISTS(SELECT 1 FROM messages WHERE conversation_id = ?) OR
                EXISTS(SELECT 1 FROM turns WHERE conversation_id = ?)""",
                (conversation_id, conversation_id)).fetchone()[0]
            active = AgentMode(conversation["agent_mode"]).roles
            for role, runtime in runtimes.items():
                old = connection.execute("SELECT session_scope FROM role_settings WHERE conversation_id = ? AND role = ?",
                                         (conversation_id, role.value)).fetchone()
                if started and role in active and (old is None or old["session_scope"] != runtime.session_scope):
                    raise ValueError("active role provider scope is locked after conversation starts")
                self._write_runtime(connection, conversation_id, role, runtime)
                if old is not None and old["session_scope"] != runtime.session_scope:
                    connection.execute("DELETE FROM agent_sessions WHERE conversation_id = ? AND role = ?",
                                       (conversation_id, role.value))
            connection.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                               (time.time(), conversation_id))

    def append(self, conversation_id: str, role: str, content: str, *,
               turn_id: str | None = None, metadata: Mapping[str, Any] | None = None,
               timestamp: float | None = None) -> int:
        with self.database.connect(write=True) as connection:
            if turn_id is not None:
                turn = connection.execute("SELECT conversation_id FROM turns WHERE id = ?", (turn_id,)).fetchone()
                if turn is None or turn["conversation_id"] != conversation_id:
                    raise ValueError("message turn does not belong to conversation")
            return insert_message(connection, conversation_id, role, content,
                                  timestamp=time.time() if timestamp is None else timestamp,
                                  turn_id=turn_id, metadata=metadata or {})

    def delete(self, conversation_id: str) -> bool:
        with self.database.connect(write=True) as connection:
            return connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount == 1

    def apply_generated_title(self, conversation_id: str, title: str) -> bool:
        """Set a generated title only while its naming state remains pending."""
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("conversation title must not be empty")
        with self.database.connect(write=True) as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title = ?, naming_state = 'generated', updated_at = ? "
                "WHERE id = ? AND naming_state = 'pending'",
                (clean_title, time.time(), conversation_id),
            )
            return cursor.rowcount == 1

    def rename(self, conversation_id: str, title: str) -> None:
        """Set a reader-chosen title; `manual` keeps automatic naming from replacing it."""
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("conversation title must not be empty")
        # `updated_at` is left alone: it orders conversations by activity, and a
        # rename is not activity.
        with self.database.connect(write=True) as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title = ?, naming_state = 'manual' WHERE id = ?",
                (clean_title, conversation_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(conversation_id)

    def messages(self, conversation_id: str, *, limit: int | None = None, offset: int = 0) -> tuple[Message, ...]:
        if offset < 0 or (limit is not None and limit < 1):
            raise ValueError("invalid message page")
        with self.database.connect() as connection:
            rows = connection.execute("""SELECT id, role, content, ts, metadata_json, turn_id
                FROM messages WHERE conversation_id = ? ORDER BY id LIMIT ? OFFSET ?""",
                (conversation_id, -1 if limit is None else limit, offset)).fetchall()
        return tuple(read_message(row) for row in rows)
