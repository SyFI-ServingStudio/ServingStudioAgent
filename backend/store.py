"""Workspace-scoped persistent state.

The browser, headless Agent API, Launcher registration bridge, and Analyzer
catalog all share the same stable workspace identity.  Runtime state therefore
belongs under ``agent-workspaces/<workspace-id>`` rather than under the
conversation-serving application.

Each workspace owns one SQLite database.  ``workspace.json`` is deliberately
small and human-readable; derived collections such as conversations and
experiments live only in SQLite.  ``registry.json`` is the safe discovery
surface consumed by the Analyzer.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

WORKSPACE_ID_PATTERN = re.compile(r"^w_[a-zA-Z0-9_-]{1,64}$")
SCHEMA_VERSION = 1
DATABASE_SCHEMA_VERSION = 2
NAMING_STATES = {"pending", "generated", "manual"}


def default_workspaces_root() -> Path:
    configured = os.environ.get("VIBESIM_WORKSPACES_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "agent-workspaces"


def default_main_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "main"


class WorkspaceRegistry:
    """Own workspace descriptors and the Analyzer-facing root registry."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        main_dir: Path | None = None,
    ) -> None:
        self.root = (root or default_workspaces_root()).resolve()
        self.main_dir = (main_dir or default_main_dir()).resolve()
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ensure_main()

    @property
    def registry_path(self) -> Path:
        return self.root / "registry.json"

    def workspace_dir(self, workspace_id: str) -> Path:
        self._validate_workspace_id(workspace_id)
        return self.root / workspace_id

    def descriptor_path(self, workspace_id: str) -> Path:
        return self.workspace_dir(workspace_id) / "workspace.json"

    def database_path(self, workspace_id: str) -> Path:
        return self.workspace_dir(workspace_id) / "workspace.sqlite"

    def ensure_main(self) -> dict[str, Any]:
        with self._lock:
            descriptor_path = self.root / "w_main" / "workspace.json"
            if not descriptor_path.is_file():
                now = time.time()
                descriptor = {
                    "schema_version": SCHEMA_VERSION,
                    "workspace_id": "w_main",
                    "display_name": "Main",
                    "naming_state": "manual",
                    "state": "active",
                    "storage_kind": "external",
                    "repo_path": os.path.relpath(self.main_dir, descriptor_path.parent),
                    "logs_path": os.path.relpath(
                        self.main_dir / "logs", descriptor_path.parent
                    ),
                    "created_at": now,
                    "last_accessed_at": now,
                    "base_workspace_id": None,
                    "base_revision": self._main_revision(),
                }
                self._write_json_atomic(descriptor_path, descriptor)
            descriptor = self.get("w_main")
            self._rewrite_registry()
            return descriptor

    def create(
        self,
        display_name: str,
        *,
        workspace_id: str | None = None,
        base_workspace_id: str = "w_main",
        base_revision: str | None = None,
        naming_state: str = "manual",
    ) -> dict[str, Any]:
        clean_name = display_name.strip()
        if not clean_name:
            raise ValueError("workspace display_name must not be empty")
        workspace_id = workspace_id or f"w_{uuid.uuid4().hex[:12]}"
        self._validate_workspace_id(workspace_id)
        self._validate_naming_state(naming_state)
        with self._lock:
            workspace_dir = self.workspace_dir(workspace_id)
            if workspace_dir.exists():
                raise ValueError(f"workspace already exists: {workspace_id}")
            now = time.time()
            descriptor = {
                "schema_version": SCHEMA_VERSION,
                "workspace_id": workspace_id,
                "display_name": clean_name,
                "naming_state": naming_state,
                "state": "active",
                "storage_kind": "managed",
                "repo_path": "repo",
                "logs_path": "repo/logs",
                "created_at": now,
                "last_accessed_at": now,
                "base_workspace_id": base_workspace_id,
                "base_revision": base_revision or self._main_revision(),
            }
            self._write_json_atomic(workspace_dir / "workspace.json", descriptor)
            self._rewrite_registry()
            return descriptor

    def list(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            descriptors: list[dict[str, Any]] = []
            for path in sorted(self.root.glob("w_*/workspace.json")):
                try:
                    descriptor = self._read_descriptor(path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if include_archived or descriptor["state"] == "active":
                    descriptors.append(descriptor)
            descriptors.sort(
                key=lambda item: (
                    item["workspace_id"] != "w_main",
                    -float(item.get("last_accessed_at", 0)),
                    item["workspace_id"],
                )
            )
            return descriptors

    def get(self, workspace_id: str) -> dict[str, Any]:
        path = self.descriptor_path(workspace_id)
        if not path.is_file():
            raise KeyError(workspace_id)
        return self._read_descriptor(path)

    def update(
        self,
        workspace_id: str,
        *,
        display_name: str | None = None,
        state: str | None = None,
        touch: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            descriptor = self.get(workspace_id)
            if display_name is not None:
                clean_name = display_name.strip()
                if not clean_name:
                    raise ValueError("workspace display_name must not be empty")
                descriptor["display_name"] = clean_name
                # An explicit API rename always wins over background generation.
                descriptor["naming_state"] = "manual"
            if state is not None:
                if state not in {"active", "archived"}:
                    raise ValueError("workspace state must be active or archived")
                if workspace_id == "w_main" and state != "active":
                    raise ValueError("w_main cannot be archived")
                descriptor["state"] = state
            if touch:
                descriptor["last_accessed_at"] = time.time()
            self._write_json_atomic(self.descriptor_path(workspace_id), descriptor)
            self._rewrite_registry()
            return descriptor

    def apply_generated_name(self, workspace_id: str, display_name: str) -> bool:
        """Set an automatic name only while the descriptor is still pending."""

        clean_name = display_name.strip()
        if not clean_name:
            raise ValueError("workspace display_name must not be empty")
        with self._lock:
            descriptor = self.get(workspace_id)
            if descriptor["naming_state"] != "pending":
                return False
            descriptor["display_name"] = clean_name
            descriptor["naming_state"] = "generated"
            descriptor["last_accessed_at"] = time.time()
            self._write_json_atomic(self.descriptor_path(workspace_id), descriptor)
            self._rewrite_registry()
            return True

    def repo_path(self, workspace_id: str) -> Path:
        descriptor_path = self.descriptor_path(workspace_id)
        descriptor = self.get(workspace_id)
        return self._resolve_descriptor_path(descriptor_path, descriptor["repo_path"])

    def logs_path(self, workspace_id: str) -> Path:
        descriptor_path = self.descriptor_path(workspace_id)
        descriptor = self.get(workspace_id)
        return self._resolve_descriptor_path(descriptor_path, descriptor["logs_path"])

    def codex_home(self, workspace_id: str, conversation_id: str) -> Path:
        self._validate_conversation_id(conversation_id)
        return self.workspace_dir(workspace_id) / "codex" / conversation_id

    def jobs_dir(self, workspace_id: str) -> Path:
        return self.workspace_dir(workspace_id) / "jobs"

    def _rewrite_registry(self) -> None:
        workspaces = []
        for descriptor in self.list(include_archived=True):
            logs_path = self.logs_path(descriptor["workspace_id"])
            workspaces.append(
                {
                    "workspace_id": descriptor["workspace_id"],
                    "display_name": descriptor["display_name"],
                    "state": descriptor["state"],
                    "logs_root": os.path.relpath(logs_path, self.root),
                }
            )
        payload = {"schema_version": SCHEMA_VERSION, "workspaces": workspaces}
        self._write_json_atomic(self.registry_path, payload)

    def _read_descriptor(self, path: Path) -> dict[str, Any]:
        descriptor = json.loads(path.read_text("utf-8"))
        required = {
            "schema_version",
            "workspace_id",
            "display_name",
            "state",
            "storage_kind",
            "repo_path",
            "logs_path",
            "created_at",
            "last_accessed_at",
        }
        missing = required - descriptor.keys()
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        if descriptor["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"{path} has unsupported schema_version {descriptor['schema_version']!r}"
            )
        self._validate_workspace_id(descriptor["workspace_id"])
        if descriptor["state"] not in {"active", "archived"}:
            raise ValueError(f"{path} has invalid state")
        if descriptor["storage_kind"] not in {"external", "managed"}:
            raise ValueError(f"{path} has invalid storage_kind")
        descriptor.setdefault("naming_state", "manual")
        self._validate_naming_state(descriptor["naming_state"])
        return descriptor

    @staticmethod
    def _resolve_descriptor_path(descriptor_path: Path, configured: str) -> Path:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = descriptor_path.parent / candidate
        return candidate.resolve()

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            "utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _validate_workspace_id(workspace_id: str) -> None:
        if not WORKSPACE_ID_PATTERN.fullmatch(workspace_id):
            raise ValueError(f"invalid workspace id: {workspace_id!r}")

    @staticmethod
    def _validate_conversation_id(conversation_id: str) -> None:
        if (
            not conversation_id
            or len(conversation_id) > 80
            or "/" in conversation_id
            or "\\" in conversation_id
            or conversation_id in {".", ".."}
        ):
            raise ValueError("invalid conversation id")

    @staticmethod
    def _validate_naming_state(naming_state: str) -> None:
        if naming_state not in NAMING_STATES:
            raise ValueError(f"invalid naming_state: {naming_state!r}")

    def _main_revision(self) -> str | None:
        head_path = self.main_dir / ".git" / "HEAD"
        try:
            head = head_path.read_text("utf-8").strip()
            if head.startswith("ref: "):
                reference = self.main_dir / ".git" / head[5:]
                return reference.read_text("utf-8").strip()
            return head
        except OSError:
            return None


class Store:
    """SQLite-backed conversation and provenance store, scoped by workspace."""

    def __init__(self, registry: WorkspaceRegistry | None = None) -> None:
        self.registry = registry or WorkspaceRegistry()
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.RLock()
        for descriptor in self.registry.list(include_archived=True):
            self._initialize_database(descriptor["workspace_id"])

    def _lock_for(self, workspace_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(workspace_id, threading.RLock())

    @contextmanager
    def _connect(self, workspace_id: str) -> Iterator[sqlite3.Connection]:
        self.registry.get(workspace_id)
        self._initialize_database(workspace_id)
        connection = sqlite3.connect(
            self.registry.database_path(workspace_id),
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_database(self, workspace_id: str) -> None:
        database_path = self.registry.database_path(workspace_id)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_for(workspace_id):
            connection = sqlite3.connect(database_path)
            try:
                connection.executescript(
                    """
                    PRAGMA journal_mode = WAL;
                    PRAGMA foreign_keys = ON;
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        naming_state TEXT NOT NULL DEFAULT 'manual',
                        sandbox TEXT NOT NULL,
                        autonomous INTEGER NOT NULL,
                        prompt_fingerprint TEXT,
                        peer_workspace TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        ts REAL NOT NULL,
                        metadata_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS messages_conversation_order
                        ON messages(conversation_id, id);
                    CREATE TABLE IF NOT EXISTS codex_sessions (
                        conversation_id TEXT NOT NULL REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        PRIMARY KEY (conversation_id, role)
                    );
                    CREATE TABLE IF NOT EXISTS turns (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        status TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS turn_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
                        sequence INTEGER NOT NULL,
                        kind TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        UNIQUE(turn_id, sequence)
                    );
                    CREATE TABLE IF NOT EXISTS execution_jobs (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        turn_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        status TEXT NOT NULL,
                        experiment_id TEXT,
                        experiment_path TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS experiments (
                        id TEXT PRIMARY KEY,
                        relative_path TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL,
                        origin_kind TEXT NOT NULL,
                        job_id TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS conversation_experiments (
                        conversation_id TEXT NOT NULL REFERENCES conversations(id)
                            ON DELETE CASCADE,
                        experiment_id TEXT NOT NULL REFERENCES experiments(id)
                            ON DELETE CASCADE,
                        turn_id TEXT,
                        relation TEXT NOT NULL,
                        PRIMARY KEY (conversation_id, experiment_id, relation)
                    );
                    """
                )
                conversation_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(conversations)"
                    ).fetchall()
                }
                if "naming_state" not in conversation_columns:
                    # Existing databases are user-owned history. They remain
                    # manual and are never silently enrolled in auto naming.
                    connection.execute(
                        """
                        ALTER TABLE conversations
                        ADD COLUMN naming_state TEXT NOT NULL DEFAULT 'manual'
                        """
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, time.time()),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (DATABASE_SCHEMA_VERSION, time.time()),
                )
                connection.commit()
            finally:
                connection.close()

    def list(self, workspace_id: str) -> list[dict[str, Any]]:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            rows = connection.execute(
                """
                SELECT id, title, naming_state, updated_at
                FROM conversations
                ORDER BY updated_at DESC, id
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def list_all(self) -> list[dict[str, Any]]:
        """Flatten active-workspace summaries without erasing their identity."""

        conversations = [
            {**conversation, "workspace_id": descriptor["workspace_id"]}
            for descriptor in self.registry.list()
            for conversation in self.list(descriptor["workspace_id"])
        ]
        conversations.sort(
            key=lambda conversation: (
                -float(conversation.get("updated_at", 0)),
                str(conversation["workspace_id"]),
                str(conversation["id"]),
            )
        )
        return conversations

    def get(self, workspace_id: str, conversation_id: str) -> dict[str, Any] | None:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            return self._conversation_payload(connection, row)

    def get_message_page(
        self,
        workspace_id: str,
        conversation_id: str,
        *,
        before: int | None,
        limit: int,
    ) -> dict[str, Any] | None:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            )
            end_index = total if before is None else min(before, total)
            start_index = max(0, end_index - limit)
            message_rows = connection.execute(
                """
                SELECT role, content, ts, metadata_json
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id
                LIMIT ? OFFSET ?
                """,
                (conversation_id, end_index - start_index, start_index),
            ).fetchall()
            payload = self._conversation_payload(
                connection,
                row,
                message_rows=message_rows,
            )
            payload["message_page"] = {
                "start_index": start_index,
                "end_index": end_index,
                "total_messages": total,
                "has_more": start_index > 0,
            }
            return payload

    def create(
        self,
        workspace_id: str,
        conversation_id: str,
        sandbox: str,
        prompt_fingerprint: str | None = None,
        *,
        autonomous: bool = False,
        peer_workspace: str | None = None,
        created_at: float | None = None,
        updated_at: float | None = None,
        title: str = "New chat",
        naming_state: str = "manual",
    ) -> dict[str, Any]:
        WorkspaceRegistry._validate_naming_state(naming_state)
        creation_time = created_at if created_at is not None else time.time()
        modification_time = updated_at if updated_at is not None else creation_time
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            connection.execute(
                """
                INSERT INTO conversations(
                    id, title, naming_state, sandbox, autonomous, prompt_fingerprint,
                    peer_workspace, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title,
                    naming_state,
                    sandbox,
                    int(autonomous),
                    prompt_fingerprint,
                    peer_workspace,
                    creation_time,
                    modification_time,
                ),
            )
        self.registry.update(workspace_id, touch=True)
        created = self.get(workspace_id, conversation_id)
        assert created is not None
        return created

    def conversation_naming_state(
        self, workspace_id: str, conversation_id: str
    ) -> str:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            row = connection.execute(
                "SELECT naming_state FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(conversation_id)
            return str(row["naming_state"])

    def apply_generated_conversation_title(
        self,
        workspace_id: str,
        conversation_id: str,
        title: str,
    ) -> bool:
        """Compare-and-set a generated title without overriding manual state."""

        clean_title = title.strip()
        if not clean_title:
            raise ValueError("conversation title must not be empty")
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET title = ?, naming_state = 'generated', updated_at = ?
                WHERE id = ? AND naming_state = 'pending'
                """,
                (clean_title, time.time(), conversation_id),
            )
            return cursor.rowcount == 1

    def update_runtime_settings(
        self,
        workspace_id: str,
        conversation_id: str,
        *,
        sandbox: str,
        autonomous: bool,
    ) -> None:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            connection.execute(
                """
                UPDATE conversations
                SET sandbox = ?, autonomous = ?, updated_at = ?
                WHERE id = ?
                """,
                (sandbox, int(autonomous), time.time(), conversation_id),
            )

    def add_message(
        self,
        workspace_id: str,
        conversation_id: str,
        role: str,
        content: str,
        *,
        ts: float | None = None,
        **metadata: Any,
    ) -> None:
        message_time = ts if ts is not None else time.time()
        filtered_metadata = {
            key: value for key, value in metadata.items() if value is not None
        }
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            connection.execute(
                """
                INSERT INTO messages(conversation_id, role, content, ts, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    message_time,
                    json.dumps(filtered_metadata, ensure_ascii=False),
                ),
            )
            if role == "user":
                row = connection.execute(
                    "SELECT title FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if row is not None and row["title"] == "New chat":
                    first_line = content.strip().splitlines()[0] if content.strip() else ""
                    connection.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        ((first_line[:48] or "New chat"), conversation_id),
                    )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (message_time, conversation_id),
            )

    def sessions_for_prompt(
        self,
        workspace_id: str,
        conversation_id: str,
        prompt_fingerprint: str,
    ) -> dict[str, str]:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            row = connection.execute(
                "SELECT prompt_fingerprint FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return {}
            if row["prompt_fingerprint"] != prompt_fingerprint:
                connection.execute(
                    "UPDATE conversations SET prompt_fingerprint = ? WHERE id = ?",
                    (prompt_fingerprint, conversation_id),
                )
                connection.execute(
                    "DELETE FROM codex_sessions WHERE conversation_id = ?",
                    (conversation_id,),
                )
                return {}
            rows = connection.execute(
                "SELECT role, session_id FROM codex_sessions WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchall()
            return {row["role"]: row["session_id"] for row in rows}

    def set_codex_session(
        self,
        workspace_id: str,
        conversation_id: str,
        role: str,
        session_id: str | None,
    ) -> None:
        if not role or not session_id:
            return
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            connection.execute(
                """
                INSERT INTO codex_sessions(conversation_id, role, session_id)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id, role)
                DO UPDATE SET session_id = excluded.session_id
                """,
                (conversation_id, role, session_id),
            )

    def delete(self, workspace_id: str, conversation_id: str) -> None:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )

    def start_turn(
        self, workspace_id: str, conversation_id: str, turn_id: str
    ) -> None:
        now = time.time()
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            connection.execute(
                """
                INSERT INTO turns(id, conversation_id, status, created_at, updated_at)
                VALUES (?, ?, 'running', ?, ?)
                """,
                (turn_id, conversation_id, now, now),
            )

    def finish_turn(self, workspace_id: str, turn_id: str, status: str) -> None:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            connection.execute(
                "UPDATE turns SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), turn_id),
            )

    def append_turn_event(
        self,
        workspace_id: str,
        turn_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            sequence = int(
                connection.execute(
                    "SELECT COUNT(*) FROM turn_events WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO turn_events(turn_id, sequence, kind, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    turn_id,
                    sequence,
                    kind,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def list_turn_events(
        self,
        workspace_id: str,
        turn_id: str,
        *,
        kinds: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return persisted events in their original turn order.

        Managed Launcher callbacks arrive through separate HTTP requests, so
        they cannot rely on the in-memory browser stream as the durable story.
        The assistant-message projection uses this ordered log when the turn
        finishes, which keeps generated experiments visible after a reload.
        """
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            rows = connection.execute(
                """
                SELECT sequence, kind, payload_json
                FROM turn_events
                WHERE turn_id = ?
                ORDER BY sequence
                """,
                (turn_id,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            kind = str(row["kind"])
            if kinds is not None and kind not in kinds:
                continue
            payload = json.loads(row["payload_json"])
            if not isinstance(payload, dict):
                continue
            events.append(
                {
                    "sequence": int(row["sequence"]),
                    "kind": kind,
                    "payload": payload,
                }
            )
        return events

    def create_job(
        self,
        workspace_id: str,
        *,
        conversation_id: str,
        turn_id: str,
        role: str,
        experiment_id: str,
        experiment_path: str,
    ) -> dict[str, Any]:
        job_id = f"j_{uuid.uuid4().hex}"
        now = time.time()
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            existing_experiment = connection.execute(
                "SELECT id FROM experiments WHERE relative_path = ?",
                (experiment_path,),
            ).fetchone()
            stable_experiment_id = (
                existing_experiment["id"]
                if existing_experiment is not None
                else experiment_id
            )
            connection.execute(
                """
                INSERT INTO execution_jobs(
                    id, conversation_id, turn_id, role, status,
                    experiment_id, experiment_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'requested', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    conversation_id,
                    turn_id,
                    role,
                    stable_experiment_id,
                    experiment_path,
                    now,
                    now,
                ),
            )
            if existing_experiment is None:
                connection.execute(
                    """
                    INSERT INTO experiments(
                        id, relative_path, status, origin_kind, job_id, created_at, updated_at
                    ) VALUES (?, ?, 'requested', 'managed', ?, ?, ?)
                    """,
                    (stable_experiment_id, experiment_path, job_id, now, now),
                )
            else:
                connection.execute(
                    """
                    UPDATE experiments
                    SET status = 'requested', job_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (job_id, now, stable_experiment_id),
                )
            connection.execute(
                """
                INSERT INTO conversation_experiments(
                    conversation_id, experiment_id, turn_id, relation
                ) VALUES (?, ?, ?, 'produced')
                ON CONFLICT(conversation_id, experiment_id, relation)
                DO UPDATE SET turn_id = excluded.turn_id
                """,
                (conversation_id, stable_experiment_id, turn_id),
            )
        return {
            "job_id": job_id,
            "experiment_id": stable_experiment_id,
            "status": "requested",
            "experiment_path": experiment_path,
        }

    def experiment_by_path(
        self,
        workspace_id: str,
        experiment_path: str,
    ) -> dict[str, Any] | None:
        """Return the workspace-local experiment identity for one log path.

        Managed registration uses this before creating a new job so the SQLite
        relationship and immutable ``experiment.meta.json`` can never choose
        different stable ids for a rerun.
        """
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            row = connection.execute(
                """
                SELECT id, relative_path, status, origin_kind, job_id,
                       created_at, updated_at
                FROM experiments
                WHERE relative_path = ?
                """,
                (experiment_path,),
            ).fetchone()
        return dict(row) if row is not None else None

    def update_job(
        self,
        workspace_id: str,
        job_id: str,
        *,
        status: str,
        conversation_id: str | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any] | None:
        now = time.time()
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            row = connection.execute(
                "SELECT * FROM execution_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            if (
                conversation_id is not None
                and row["conversation_id"] != conversation_id
            ) or (turn_id is not None and row["turn_id"] != turn_id):
                return None
            connection.execute(
                "UPDATE execution_jobs SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, job_id),
            )
            if row["experiment_id"]:
                connection.execute(
                    "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now, row["experiment_id"]),
                )
            return {
                "job_id": job_id,
                "experiment_id": row["experiment_id"],
                "conversation_id": row["conversation_id"],
                "turn_id": row["turn_id"],
                "status": status,
                "experiment_path": row["experiment_path"],
            }

    def list_experiments(
        self, workspace_id: str, *, conversation_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            if conversation_id:
                rows = connection.execute(
                    """
                    SELECT e.*, ce.conversation_id, ce.turn_id, ce.relation
                    FROM experiments e
                    JOIN conversation_experiments ce ON ce.experiment_id = e.id
                    WHERE ce.conversation_id = ?
                    ORDER BY e.updated_at DESC
                    """,
                    (conversation_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT e.*, ce.conversation_id, ce.turn_id, ce.relation
                    FROM experiments e
                    LEFT JOIN conversation_experiments ce ON ce.experiment_id = e.id
                    ORDER BY e.updated_at DESC
                    """
                ).fetchall()
            return [dict(row) for row in rows]

    def import_conversation(
        self,
        workspace_id: str,
        conversation: dict[str, Any],
    ) -> None:
        conversation_id = str(conversation["id"])
        if self.get(workspace_id, conversation_id) is not None:
            return
        source_created_at = (
            float(conversation["created_at"])
            if conversation.get("created_at") is not None
            else time.time()
        )
        source_updated_at = (
            float(conversation["updated_at"])
            if conversation.get("updated_at") is not None
            else source_created_at
        )
        self.create(
            workspace_id,
            conversation_id,
            conversation.get("sandbox", "workspace-write"),
            conversation.get("prompt_fingerprint"),
            autonomous=bool(conversation.get("autonomous", False)),
            peer_workspace=conversation.get("peer_workspace"),
            created_at=source_created_at,
            updated_at=source_updated_at,
            title=conversation.get("title") or "New chat",
        )
        for message in conversation.get("messages") or []:
            metadata = {
                key: value
                for key, value in message.items()
                if key not in {"role", "content", "ts"}
            }
            self.add_message(
                workspace_id,
                conversation_id,
                message.get("role", "assistant"),
                message.get("content", ""),
                ts=float(message.get("ts") or time.time()),
                **metadata,
            )
        for role, session_id in (conversation.get("codex_sessions") or {}).items():
            self.set_codex_session(
                workspace_id,
                conversation_id,
                role,
                session_id,
            )
        self.restore_imported_timestamps(
            workspace_id,
            conversation_id,
            created_at=source_created_at,
            updated_at=source_updated_at,
        )

    def restore_imported_timestamps(
        self,
        workspace_id: str,
        conversation_id: str,
        *,
        created_at: float,
        updated_at: float,
    ) -> None:
        """Restore source timestamps after migration replays append-only messages.

        ``add_message`` normally advances ``updated_at``. A lossless import must
        replay message rows and then put the source conversation timestamps back
        exactly so history ordering does not drift by a few microseconds.
        """
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET created_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (created_at, updated_at, conversation_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(conversation_id)

    @staticmethod
    def _message_payload(row: sqlite3.Row) -> dict[str, Any]:
        payload = {
            "role": row["role"],
            "content": row["content"],
            "ts": row["ts"],
        }
        try:
            payload.update(json.loads(row["metadata_json"]))
        except (TypeError, json.JSONDecodeError):
            pass
        return payload

    def _conversation_payload(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        message_rows: list[sqlite3.Row] | None = None,
    ) -> dict[str, Any]:
        conversation_id = row["id"]
        if message_rows is None:
            message_rows = connection.execute(
                """
                SELECT role, content, ts, metadata_json
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id
                """,
                (conversation_id,),
            ).fetchall()
        session_rows = connection.execute(
            "SELECT role, session_id FROM codex_sessions WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        return {
            "id": conversation_id,
            "title": row["title"],
            "naming_state": row["naming_state"],
            "sandbox": row["sandbox"],
            "autonomous": bool(row["autonomous"]),
            "prompt_fingerprint": row["prompt_fingerprint"],
            "peer_workspace": row["peer_workspace"],
            "codex_sessions": {
                session["role"]: session["session_id"] for session in session_rows
            },
            "messages": [self._message_payload(message) for message in message_rows],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
