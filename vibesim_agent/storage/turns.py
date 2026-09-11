"""Ordered turn events and atomic terminal persistence."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Literal

from .conversations import insert_message
from .database import Database


def _insert_event(connection: sqlite3.Connection, turn_id: str, kind: str,
                  payload: dict[str, Any]) -> int:
    sequence = connection.execute(
        "SELECT COALESCE(MAX(sequence), -1) + 1 FROM turn_events WHERE turn_id = ?",
        (turn_id,),
    ).fetchone()[0]
    connection.execute(
        "INSERT INTO turn_events(turn_id, sequence, kind, payload_json) VALUES (?, ?, ?, ?)",
        (turn_id, sequence, kind, json.dumps(payload, ensure_ascii=False)),
    )
    return sequence


class Turns:
    def __init__(self, database: Database):
        self.database = database

    def running(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM turns WHERE status = 'running' ORDER BY created_at, id"
            ).fetchall()
        return [dict(row) for row in rows]

    def interrupt(self, conversation_id: str, turn_id: str, *, text: str,
                  metadata: dict[str, Any]) -> bool:
        """Recover an orphan without changing its recorded events or resume state."""
        now = time.time()
        with self.database.connect(write=True) as connection:
            claimed = connection.execute(
                "UPDATE turns SET status = 'interrupted', updated_at = ? "
                "WHERE id = ? AND conversation_id = ? AND status = 'running'",
                (now, turn_id, conversation_id),
            )
            if claimed.rowcount != 1:
                return False
            insert_message(connection, conversation_id, "assistant", text,
                           timestamp=now, turn_id=turn_id, metadata=metadata)
        return True

    def list(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute("""SELECT turns.id AS turn_id, turns.status, turns.created_at,
                turns.updated_at, COUNT(turn_events.id) AS event_count FROM turns
                LEFT JOIN turn_events ON turn_events.turn_id = turns.id
                WHERE turns.conversation_id = ? GROUP BY turns.id ORDER BY turns.created_at, turns.id""",
                (conversation_id,)).fetchall()
        return [dict(row) for row in rows]

    def get(self, conversation_id: str, turn_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM turns WHERE id = ? AND conversation_id = ?",
                (turn_id, conversation_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def active(self, conversation_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM turns WHERE conversation_id = ? AND status = 'running' "
                "ORDER BY created_at DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def start(self, conversation_id: str, turn_id: str, user_text: str,
              metadata: dict[str, Any] | None = None, *, settings: dict[str, Any] | None = None) -> int:
        if settings is not None and set(settings) != {"sandbox", "autonomous", "agent_mode", "interrupted_role", "prompt_fingerprint"}:
            raise ValueError("invalid turn settings fields")
        now = time.time()
        with self.database.connect(write=True) as connection:
            if settings is not None:
                connection.execute("""UPDATE conversations SET sandbox = ?, autonomous = ?, agent_mode = ?,
                    interrupted_role = ?, prompt_fingerprint = ? WHERE id = ?""",
                    (settings["sandbox"], settings["autonomous"], settings["agent_mode"],
                     settings["interrupted_role"], settings["prompt_fingerprint"], conversation_id))
            connection.execute("""INSERT INTO turns(id, conversation_id, status, created_at, updated_at)
                VALUES (?, ?, 'running', ?, ?)""", (turn_id, conversation_id, now, now))
            return insert_message(connection, conversation_id, "user", user_text,
                                  timestamp=now, turn_id=turn_id, metadata=metadata or {})

    def append_event(self, turn_id: str, kind: str, payload: dict[str, Any]) -> int:
        with self.database.connect(write=True) as connection:
            turn = connection.execute("SELECT status FROM turns WHERE id = ?", (turn_id,)).fetchone()
            if turn is None:
                raise KeyError(turn_id)
            if turn["status"] != "running":
                raise ValueError("cannot append to a terminal turn")
            return _insert_event(connection, turn_id, kind, payload)

    def events(self, conversation_id: str, turn_id: str, *,
               after_sequence: int = -1) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            owner = connection.execute("SELECT conversation_id FROM turns WHERE id = ?", (turn_id,)).fetchone()
            if owner is None or owner["conversation_id"] != conversation_id:
                raise KeyError(turn_id)
            rows = connection.execute("SELECT sequence, kind, payload_json FROM turn_events "
                                      "WHERE turn_id = ? AND sequence > ? ORDER BY sequence",
                                      (turn_id, after_sequence)).fetchall()
        return [{"sequence": row["sequence"], "kind": row["kind"], "payload": json.loads(row["payload_json"])} for row in rows]

    def finish(self, conversation_id: str, turn_id: str, *, text: str,
               status: Literal["complete", "failed"], interrupted_role: str | None = None,
               metadata: dict[str, Any] | None = None,
               terminal_event: dict[str, Any] | None = None) -> bool:
        """Commit answer and optional done together.

        interrupted_role=None preserves the resume role; an empty string clears it.
        """
        if status not in {"complete", "failed"}:
            raise ValueError("invalid terminal status")
        now = time.time()
        with self.database.connect(write=True) as connection:
            row = connection.execute("SELECT conversation_id, status FROM turns WHERE id = ?", (turn_id,)).fetchone()
            if row is None or row["conversation_id"] != conversation_id:
                raise KeyError(turn_id)
            if row["status"] != "running":
                return False
            insert_message(connection, conversation_id, "assistant", text, timestamp=now,
                           turn_id=turn_id, metadata=metadata or {})
            if interrupted_role is not None:
                connection.execute("UPDATE conversations SET interrupted_role = ? WHERE id = ?",
                                   (interrupted_role, conversation_id))
            connection.execute("UPDATE turns SET status = ?, updated_at = ? WHERE id = ?",
                               (status, now, turn_id))
            if terminal_event is not None:
                _insert_event(connection, turn_id, "done", terminal_event)
        return True
