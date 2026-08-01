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
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .codex_runtime.config import (
    CODEX_FAMILIES,
    DEFAULT_CODEX_EFFORT,
    DEFAULT_CODEX_FAMILY,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_SERVICE_TIER,
    LEGACY_BACKEND_MODELS,
    codex_model,
    normalize_role_runtime,
)

WORKSPACE_ID_PATTERN = re.compile(r"^w_[a-zA-Z0-9_-]{1,64}$")
SCHEMA_VERSION = 1
DATABASE_SCHEMA_VERSION = 6
NAMING_STATES = {"pending", "generated", "manual"}


def _normalized_runtime(runtime: dict[str, str] | None) -> dict[str, str]:
    return normalize_role_runtime(
        (runtime or {}).get("model"),
        (runtime or {}).get("effort"),
        (runtime or {}).get("service_tier") or (runtime or {}).get("serviceTier"),
    )


def _imported_runtime(conversation: dict[str, Any], role: str) -> dict[str, str]:
    """Read one role's runtime from an exported conversation, old shape or new.

    Legacy exports carry ``codex_backends: {role: "traditional"}``; the backend
    id resolves to that family's default model.
    """
    runtime = (conversation.get("codex_runtime") or {}).get(role)
    if isinstance(runtime, dict):
        return _normalized_runtime(runtime)
    legacy = (conversation.get("codex_backends") or {}).get(role)
    return _normalized_runtime({"model": legacy} if legacy else None)


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

    def discard_failed_creation(self, workspace_id: str) -> None:
        """Remove state created by a workspace request that did not complete.

        This is intentionally narrower than a workspace-delete API: callers
        may use it only while unwinding the same create request, before the
        workspace has been returned to a client.
        """
        if workspace_id == "w_main":
            raise ValueError("w_main cannot be discarded")
        with self._lock:
            shutil.rmtree(self.workspace_dir(workspace_id), ignore_errors=True)
            self._rewrite_registry()

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
        default_model = DEFAULT_CODEX_MODEL
        default_effort = DEFAULT_CODEX_EFFORT
        default_service_tier = DEFAULT_CODEX_SERVICE_TIER
        default_family = DEFAULT_CODEX_FAMILY
        with self._lock_for(workspace_id):
            connection = sqlite3.connect(database_path)
            try:
                connection.executescript(
                    f"""
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
                        orchestrator_model TEXT NOT NULL DEFAULT '{default_model}',
                        orchestrator_effort TEXT NOT NULL DEFAULT '{default_effort}',
                        orchestrator_service_tier TEXT NOT NULL DEFAULT '{default_service_tier}',
                        implementer_model TEXT NOT NULL DEFAULT '{default_model}',
                        implementer_effort TEXT NOT NULL DEFAULT '{default_effort}',
                        implementer_service_tier TEXT NOT NULL DEFAULT '{default_service_tier}',
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
                        family TEXT NOT NULL DEFAULT '{default_family}',
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
                        job_kind TEXT NOT NULL DEFAULT 'simulation',
                        artifact_path TEXT,
                        resource_id TEXT,
                        analyzer_resource_id TEXT,
                        descriptor_json TEXT NOT NULL DEFAULT '{{}}',
                        summary_json TEXT,
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
                # Pre-registry databases stored one backend id per role. A role now
                # picks a model plus a reasoning effort, so widen those two columns
                # into four and carry the old ids onto their family's default model.
                for column_name, default_value in (
                    ("orchestrator_model", default_model),
                    ("orchestrator_effort", default_effort),
                    ("orchestrator_service_tier", default_service_tier),
                    ("implementer_model", default_model),
                    ("implementer_effort", default_effort),
                    ("implementer_service_tier", default_service_tier),
                ):
                    if column_name not in conversation_columns:
                        connection.execute(
                            f"ALTER TABLE conversations ADD COLUMN {column_name} "
                            f"TEXT NOT NULL DEFAULT '{default_value}'"
                        )
                for legacy_column, model_column, effort_column in (
                    (
                        "orchestrator_backend",
                        "orchestrator_model",
                        "orchestrator_effort",
                    ),
                    ("implementer_backend", "implementer_model", "implementer_effort"),
                ):
                    if legacy_column not in conversation_columns:
                        continue
                    for backend_id, model_id in LEGACY_BACKEND_MODELS.items():
                        family = CODEX_FAMILIES[codex_model(model_id).family_id]
                        connection.execute(
                            f"UPDATE conversations SET {model_column} = ?, "
                            f"{effort_column} = ? WHERE {legacy_column} = ?",
                            (model_id, family.default_effort, backend_id),
                        )
                    connection.execute(
                        f"ALTER TABLE conversations DROP COLUMN {legacy_column}"
                    )
                session_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(codex_sessions)"
                    ).fetchall()
                }
                # A session belongs to a provider profile, not to one model: the
                # column always meant "which auth home recorded this rollout", so
                # it is renamed to say so and its values move onto family ids.
                if "family" not in session_columns:
                    if "backend_id" in session_columns:
                        connection.execute(
                            "ALTER TABLE codex_sessions "
                            "RENAME COLUMN backend_id TO family"
                        )
                        for backend_id, model_id in LEGACY_BACKEND_MODELS.items():
                            connection.execute(
                                "UPDATE codex_sessions SET family = ? WHERE family = ?",
                                (codex_model(model_id).family_id, backend_id),
                            )
                    else:
                        connection.execute(
                            "ALTER TABLE codex_sessions ADD COLUMN family "
                            f"TEXT NOT NULL DEFAULT '{default_family}'"
                        )
                execution_job_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(execution_jobs)"
                    ).fetchall()
                }
                # Existing simulation jobs migrate in place. New typed jobs use
                # artifact_path/resource_id without pretending to be experiments.
                for column_name, column_definition in (
                    ("job_kind", "TEXT NOT NULL DEFAULT 'simulation'"),
                    ("artifact_path", "TEXT"),
                    ("resource_id", "TEXT"),
                    ("analyzer_resource_id", "TEXT"),
                    ("descriptor_json", "TEXT NOT NULL DEFAULT '{}'"),
                    ("summary_json", "TEXT"),
                ):
                    if column_name not in execution_job_columns:
                        connection.execute(
                            f"ALTER TABLE execution_jobs ADD COLUMN "
                            f"{column_name} {column_definition}"
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

    def list_artifact_jobs(self, workspace_id: str) -> list[dict[str, Any]]:
        """List browser-navigable non-simulation results in one workspace."""
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            rows = connection.execute(
                """
                SELECT jobs.id, jobs.conversation_id, conversations.title AS conversation_title,
                       jobs.turn_id, jobs.status, jobs.job_kind, jobs.artifact_path,
                       jobs.resource_id, jobs.analyzer_resource_id,
                       jobs.descriptor_json, jobs.summary_json,
                       jobs.created_at, jobs.updated_at
                FROM execution_jobs AS jobs
                JOIN conversations ON conversations.id = jobs.conversation_id
                WHERE jobs.job_kind != 'simulation' AND jobs.resource_id IS NOT NULL
                ORDER BY jobs.updated_at DESC, jobs.id
                """
            ).fetchall()
        return [
            {
                "job_id": row["id"],
                "conversation_id": row["conversation_id"],
                "conversation_title": row["conversation_title"],
                "turn_id": row["turn_id"],
                "status": row["status"],
                "job_kind": row["job_kind"],
                "artifact_path": row["artifact_path"],
                "resource_id": row["resource_id"],
                "analyzer_resource_id": row["analyzer_resource_id"],
                "descriptor": json.loads(row["descriptor_json"] or "{}"),
                "summary": (
                    json.loads(row["summary_json"]) if row["summary_json"] else None
                ),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def list_all_artifact_jobs(self) -> list[dict[str, Any]]:
        """Flatten typed result jobs across active workspaces for Page 0."""
        jobs = [
            {**job, "workspace_id": descriptor["workspace_id"]}
            for descriptor in self.registry.list()
            for job in self.list_artifact_jobs(descriptor["workspace_id"])
        ]
        jobs.sort(
            key=lambda job: (
                -float(job.get("updated_at", 0)),
                str(job["workspace_id"]),
                str(job["job_id"]),
            )
        )
        return jobs

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
        orchestrator_runtime: dict[str, str] | None = None,
        implementer_runtime: dict[str, str] | None = None,
        peer_workspace: str | None = None,
        created_at: float | None = None,
        updated_at: float | None = None,
        title: str = "New chat",
        naming_state: str = "manual",
    ) -> dict[str, Any]:
        WorkspaceRegistry._validate_naming_state(naming_state)
        creation_time = created_at if created_at is not None else time.time()
        modification_time = updated_at if updated_at is not None else creation_time
        orchestrator = _normalized_runtime(orchestrator_runtime)
        implementer = _normalized_runtime(implementer_runtime)
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            connection.execute(
                """
                INSERT INTO conversations(
                    id, title, naming_state, sandbox, autonomous,
                    orchestrator_model, orchestrator_effort, orchestrator_service_tier,
                    implementer_model, implementer_effort, implementer_service_tier,
                    prompt_fingerprint,
                    peer_workspace, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title,
                    naming_state,
                    sandbox,
                    int(autonomous),
                    orchestrator["model"],
                    orchestrator["effort"],
                    orchestrator["service_tier"],
                    implementer["model"],
                    implementer["effort"],
                    implementer["service_tier"],
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

    def conversation_naming_state(self, workspace_id: str, conversation_id: str) -> str:
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

    def update_codex_runtime(
        self,
        workspace_id: str,
        conversation_id: str,
        *,
        orchestrator_runtime: dict[str, str],
        implementer_runtime: dict[str, str],
    ) -> None:
        """Change a role's model, effort, and tier within resume compatibility.

        Before the conversation starts anything goes. Once it has history, a role
        may still move to a sibling model, any effort, and any supported service
        tier — all are per-call Codex options and the rollout stays resumable — but it
        may not cross families, because that rollout lives in the other family's
        ``CODEX_HOME`` under different auth.
        """
        orchestrator = _normalized_runtime(orchestrator_runtime)
        implementer = _normalized_runtime(implementer_runtime)
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            current = connection.execute(
                "SELECT orchestrator_model, implementer_model FROM conversations "
                "WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if current is None:
                raise KeyError(conversation_id)
            message_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            )
            turn_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM turns WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            )
            started = bool(message_count or turn_count)
            changed_families = [
                role
                for role, selection, stored in (
                    ("orchestrator", orchestrator, current["orchestrator_model"]),
                    ("implementer", implementer, current["implementer_model"]),
                )
                if codex_model(selection["model"]).family_id
                != codex_model(stored).family_id
            ]
            if started and changed_families:
                raise ValueError(
                    "model family is locked after the conversation starts: "
                    + ", ".join(changed_families)
                )
            connection.execute(
                """
                UPDATE conversations
                SET orchestrator_model = ?, orchestrator_effort = ?,
                    orchestrator_service_tier = ?, implementer_model = ?,
                    implementer_effort = ?, implementer_service_tier = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    orchestrator["model"],
                    orchestrator["effort"],
                    orchestrator["service_tier"],
                    implementer["model"],
                    implementer["effort"],
                    implementer["service_tier"],
                    time.time(),
                    conversation_id,
                ),
            )
            for role in changed_families:
                # Only a family change orphans a rollout; a sibling model resumes it.
                connection.execute(
                    "DELETE FROM codex_sessions WHERE conversation_id = ? AND role = ?",
                    (conversation_id, role),
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
                    first_line = (
                        content.strip().splitlines()[0] if content.strip() else ""
                    )
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
        runtimes: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, str]:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            row = connection.execute(
                "SELECT prompt_fingerprint FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return {}
            if row["prompt_fingerprint"] != prompt_fingerprint:
                # Role contracts are re-injected on every Codex call. Keep the
                # durable sessions—and therefore their conversation history—
                # when those contracts change; the fingerprint is provenance,
                # not a session compatibility boundary.
                connection.execute(
                    "UPDATE conversations SET prompt_fingerprint = ? WHERE id = ?",
                    (prompt_fingerprint, conversation_id),
                )
            rows = connection.execute(
                "SELECT role, family, session_id FROM codex_sessions "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchall()
            # A rollout is resumable by any model of the family that recorded it,
            # so the reuse test compares families, never the exact model.
            selected_families = {
                role: codex_model(selection["model"]).family_id
                for role, selection in (runtimes or {}).items()
            }
            return {
                session["role"]: session["session_id"]
                for session in rows
                if session["family"]
                == selected_families.get(session["role"], session["family"])
            }

    def set_codex_session(
        self,
        workspace_id: str,
        conversation_id: str,
        role: str,
        session_id: str | None,
        family: str = DEFAULT_CODEX_FAMILY,
    ) -> None:
        if not role or not session_id:
            return
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            connection.execute(
                """
                INSERT INTO codex_sessions(conversation_id, role, family, session_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id, role)
                DO UPDATE SET family = excluded.family,
                              session_id = excluded.session_id
                """,
                (conversation_id, role, family, session_id),
            )

    def delete(self, workspace_id: str, conversation_id: str) -> None:
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )

    def start_turn(self, workspace_id: str, conversation_id: str, turn_id: str) -> None:
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

    def list_running_turns(self, workspace_id: str) -> list[dict[str, Any]]:
        """Return turns that cannot still be owned after a backend restart."""
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, created_at, updated_at
                FROM turns
                WHERE status = 'running'
                ORDER BY created_at, id
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def interrupt_orphaned_turn(
        self,
        workspace_id: str,
        turn_id: str,
        conversation_id: str,
        *,
        content: str,
        activity: list[dict[str, Any]],
    ) -> bool:
        """Atomically preserve one orphan turn as a reloadable assistant message.

        The conditional status update is the ownership claim. It prevents a
        repeated startup recovery from appending the same timeline twice.
        """
        now = time.time()
        metadata = json.dumps({"activity": activity}, ensure_ascii=False)
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            cursor = connection.execute(
                """
                UPDATE turns
                SET status = 'interrupted', updated_at = ?
                WHERE id = ? AND conversation_id = ? AND status = 'running'
                """,
                (now, turn_id, conversation_id),
            )
            if cursor.rowcount != 1:
                return False
            connection.execute(
                """
                INSERT INTO messages(conversation_id, role, content, ts, metadata_json)
                VALUES (?, 'assistant', ?, ?, ?)
                """,
                (conversation_id, content, now, metadata),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            return True

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
            "job_kind": "simulation",
        }

    def create_artifact_job(
        self,
        workspace_id: str,
        *,
        conversation_id: str,
        turn_id: str,
        role: str,
        job_kind: str,
        artifact_path: str,
        analyzer_resource_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a typed non-simulation job linked directly to a conversation."""
        job_id = f"j_{uuid.uuid4().hex}"
        resource_id = f"jr_{uuid.uuid4().hex}"
        now = time.time()
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            connection.execute(
                """
                INSERT INTO execution_jobs(
                    id, conversation_id, turn_id, role, status,
                    job_kind, artifact_path, resource_id, analyzer_resource_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'requested', ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    conversation_id,
                    turn_id,
                    role,
                    job_kind,
                    artifact_path,
                    resource_id,
                    analyzer_resource_id,
                    now,
                    now,
                ),
            )
        return {
            "job_id": job_id,
            "resource_id": resource_id,
            "analyzer_resource_id": analyzer_resource_id,
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "status": "requested",
            "job_kind": job_kind,
            "artifact_path": artifact_path,
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

    def get_experiment(
        self,
        workspace_id: str,
        experiment_id: str,
    ) -> dict[str, Any] | None:
        """Resolve a stable Analyzer experiment inside one workspace boundary."""
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            row = connection.execute(
                """
                SELECT id, relative_path, status, origin_kind, job_id,
                       created_at, updated_at
                FROM experiments
                WHERE id = ?
                """,
                (experiment_id,),
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
                """
                UPDATE execution_jobs
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
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
                "job_kind": row["job_kind"],
                "artifact_path": row["artifact_path"],
                "resource_id": row["resource_id"],
                "analyzer_resource_id": row["analyzer_resource_id"],
            }

    def artifact_job_by_resource(
        self,
        workspace_id: str,
        resource_id: str,
    ) -> dict[str, Any] | None:
        """Resolve one non-simulation job for the browser result surface."""
        with self._lock_for(workspace_id), self._connect(workspace_id) as connection:
            row = connection.execute(
                """
                SELECT * FROM execution_jobs
                WHERE resource_id = ? AND job_kind != 'simulation'
                """,
                (resource_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["id"],
            "resource_id": row["resource_id"],
            "analyzer_resource_id": row["analyzer_resource_id"],
            "conversation_id": row["conversation_id"],
            "turn_id": row["turn_id"],
            "role": row["role"],
            "status": row["status"],
            "job_kind": row["job_kind"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
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
            orchestrator_runtime=_imported_runtime(conversation, "orchestrator"),
            implementer_runtime=_imported_runtime(conversation, "implementer"),
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
                family=codex_model(
                    _imported_runtime(conversation, role)["model"]
                ).family_id,
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
            "SELECT role, family, session_id FROM codex_sessions "
            "WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        return {
            "id": conversation_id,
            "title": row["title"],
            "naming_state": row["naming_state"],
            "sandbox": row["sandbox"],
            "autonomous": bool(row["autonomous"]),
            "codex_runtime": {
                "orchestrator": {
                    "model": row["orchestrator_model"],
                    "effort": row["orchestrator_effort"],
                    "serviceTier": row["orchestrator_service_tier"],
                },
                "implementer": {
                    "model": row["implementer_model"],
                    "effort": row["implementer_effort"],
                    "serviceTier": row["implementer_service_tier"],
                },
            },
            "prompt_fingerprint": row["prompt_fingerprint"],
            "peer_workspace": row["peer_workspace"],
            "codex_sessions": {
                session["role"]: session["session_id"] for session in session_rows
            },
            "messages": [self._message_payload(message) for message in message_rows],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
