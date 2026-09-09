import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.codex_runtime.config import CODEXDS_MODEL as DEEPSEEK_MODEL
from backend.codex_runtime.config import (
    DEFAULT_CODEX_EFFORT,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_SERVICE_TIER,
)
from backend.migrate_workspaces import (
    execute_migration,
    repair_completed_timestamps,
)
from backend.store import Store, WorkspaceRegistry
from model_catalog_fixture import install_model_catalog

# A role the conversation's mode never drives still round-trips, at defaults.
UNUSED_ROLE_RUNTIME = {
    "model": DEFAULT_CODEX_MODEL,
    "effort": DEFAULT_CODEX_EFFORT,
    "serviceTier": DEFAULT_CODEX_SERVICE_TIER,
}


def _write_legacy_conversation_database(database_path: Path) -> None:
    """One conversation in the pre-registry schema: a backend id per role."""
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                naming_state TEXT NOT NULL DEFAULT 'manual',
                sandbox TEXT NOT NULL,
                autonomous INTEGER NOT NULL,
                orchestrator_backend TEXT NOT NULL DEFAULT 'traditional',
                implementer_backend TEXT NOT NULL DEFAULT 'traditional',
                prompt_fingerprint TEXT,
                peer_workspace TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE codex_sessions (
                conversation_id TEXT NOT NULL REFERENCES conversations(id)
                    ON DELETE CASCADE,
                role TEXT NOT NULL,
                backend_id TEXT NOT NULL DEFAULT 'traditional',
                session_id TEXT NOT NULL,
                PRIMARY KEY (conversation_id, role)
            );
            INSERT INTO conversations(
                id, title, naming_state, sandbox, autonomous,
                orchestrator_backend, implementer_backend, prompt_fingerprint,
                peer_workspace, created_at, updated_at
            ) VALUES (
                'legacy', 'Legacy chat', 'manual', 'workspace-write', 0,
                'codexds', 'traditional', 'contract', NULL, 1.0, 1.0
            );
            INSERT INTO codex_sessions(conversation_id, role, backend_id, session_id)
            VALUES ('legacy', 'orchestrator', 'codexds', 'legacy-session');
            """
        )
        connection.commit()
    finally:
        connection.close()


class WorkspaceStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        install_model_catalog(self)

    def make_registry(self, temporary_directory: str) -> WorkspaceRegistry:
        root = Path(temporary_directory)
        main_dir = root / "main"
        (main_dir / "logs").mkdir(parents=True)
        return WorkspaceRegistry(root / "agent-workspaces", main_dir=main_dir)

    def test_interrupted_role_survives_reading_and_is_cleared_by_writing(self) -> None:
        """Once the user is talking to the implementer, a second question is as
        natural as the first, so reading must not end the conversation with it.
        Clearing is the turn's job, through the same setter."""
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            store = Store(registry)
            store.create("w_main", "conversation", "workspace-write")

            self.assertEqual(
                store.get("w_main", "conversation")["interrupted_role"], ""
            )
            store.set_interrupted_role("w_main", "conversation", "implementer")
            self.assertEqual(
                store.get("w_main", "conversation")["interrupted_role"], "implementer"
            )

            self.assertEqual(
                store.read_interrupted_role("w_main", "conversation"), "implementer"
            )
            self.assertEqual(
                store.read_interrupted_role("w_main", "conversation"), "implementer"
            )

            store.set_interrupted_role("w_main", "conversation", "")
            self.assertEqual(store.read_interrupted_role("w_main", "conversation"), "")
            self.assertEqual(
                store.get("w_main", "conversation")["interrupted_role"], ""
            )

    def test_pages_backwards_without_mutating_history(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            store = Store(registry)
            store.create("w_main", "conversation", "read-only")
            for message_index in range(7):
                store.add_message(
                    "w_main",
                    "conversation",
                    "user" if message_index % 2 == 0 else "assistant",
                    f"message-{message_index}",
                )

            newest_page = store.get_message_page(
                "w_main",
                "conversation",
                before=None,
                limit=3,
            )
            assert newest_page is not None
            self.assertEqual(
                [message["content"] for message in newest_page["messages"]],
                ["message-4", "message-5", "message-6"],
            )
            self.assertEqual(
                newest_page["message_page"],
                {
                    "start_index": 4,
                    "end_index": 7,
                    "total_messages": 7,
                    "has_more": True,
                },
            )

            older_page = store.get_message_page(
                "w_main",
                "conversation",
                before=newest_page["message_page"]["start_index"],
                limit=3,
            )
            assert older_page is not None
            self.assertEqual(
                [message["content"] for message in older_page["messages"]],
                ["message-1", "message-2", "message-3"],
            )
            self.assertTrue(older_page["message_page"]["has_more"])

            newest_page["messages"][0]["content"] = "changed outside store"
            full_conversation = store.get("w_main", "conversation")
            assert full_conversation is not None
            self.assertEqual(full_conversation["messages"][4]["content"], "message-4")

    def test_conversations_are_scoped_by_workspace(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            registry.create("Second", workspace_id="w_second")
            store = Store(registry)
            store.create("w_main", "same-id", "read-only", title="Main chat")
            store.create("w_second", "same-id", "workspace-write", title="Second chat")

            self.assertEqual(store.get("w_main", "same-id")["title"], "Main chat")
            self.assertEqual(store.get("w_second", "same-id")["title"], "Second chat")

    def test_prompt_change_preserves_role_sessions(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            store = Store(registry)
            store.create(
                "w_main",
                "conversation",
                "workspace-write",
                prompt_fingerprint="old-contract",
            )
            store.set_codex_session(
                "w_main",
                "conversation",
                "orchestrator",
                "orchestrator-session",
            )
            store.set_codex_session(
                "w_main",
                "conversation",
                "implementer",
                "implementer-session",
            )

            sessions = store.sessions_for_prompt(
                "w_main",
                "conversation",
                "new-contract",
            )

            self.assertEqual(
                sessions,
                {
                    "orchestrator": "orchestrator-session",
                    "implementer": "implementer-session",
                },
            )
            self.assertEqual(
                store.get("w_main", "conversation")["prompt_fingerprint"],
                "new-contract",
            )

    def test_role_runtimes_are_persisted_and_sessions_key_on_family(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            store = Store(registry)
            store.create(
                "w_main",
                "conversation",
                "workspace-write",
                role_runtimes={
                    "orchestrator": {"model": DEEPSEEK_MODEL, "effort": "high"},
                    "implementer": {"model": "gpt-5.6-sol", "effort": "xhigh"},
                },
            )
            store.set_codex_session(
                "w_main",
                "conversation",
                "orchestrator",
                "deepseek-session",
                family="deepseek",
            )

            self.assertEqual(
                store.get("w_main", "conversation")["codex_runtime"],
                {
                    "orchestrator": {
                        "model": DEEPSEEK_MODEL,
                        "effort": "high",
                        "serviceTier": "default",
                    },
                    "implementer": {
                        "model": "gpt-5.6-sol",
                        "effort": "xhigh",
                        "serviceTier": "default",
                    },
                    "assistant": UNUSED_ROLE_RUNTIME,
                },
            )
            self.assertEqual(
                store.sessions_for_prompt(
                    "w_main",
                    "conversation",
                    "contract",
                    runtimes={
                        "orchestrator": {"model": DEEPSEEK_MODEL, "effort": "max"},
                        "implementer": {"model": "gpt-5.6-sol", "effort": "xhigh"},
                    },
                ),
                {"orchestrator": "deepseek-session"},
            )
            self.assertEqual(
                store.sessions_for_prompt(
                    "w_main",
                    "conversation",
                    "contract",
                    runtimes={
                        "orchestrator": {"model": "gpt-5.6-sol", "effort": "xhigh"},
                        "implementer": {"model": "gpt-5.6-sol", "effort": "xhigh"},
                    },
                ),
                {},
            )

    def test_started_conversation_locks_family_but_not_model_or_effort(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            store = Store(registry)
            store.create(
                "w_main",
                "conversation",
                "workspace-write",
                role_runtimes={
                    "orchestrator": {"model": "gpt-5.6-sol", "effort": "xhigh"},
                    "implementer": {"model": "gpt-5.6-sol", "effort": "xhigh"},
                },
            )
            store.set_codex_session(
                "w_main", "conversation", "orchestrator", "sol-session", family="gpt"
            )
            store.add_message("w_main", "conversation", "user", "start")

            # A sibling model resumes the same rollout, so the session survives.
            store.update_codex_runtime(
                "w_main",
                "conversation",
                role_runtimes={
                    "orchestrator": {
                        "model": "gpt-5.6-luna",
                        "effort": "low",
                        "serviceTier": "fast",
                    },
                    "implementer": {"model": "gpt-5.6-sol", "effort": "max"},
                },
            )
            self.assertEqual(
                store.get("w_main", "conversation")["codex_runtime"],
                {
                    "orchestrator": {
                        "model": "gpt-5.6-luna",
                        "effort": "low",
                        "serviceTier": "fast",
                    },
                    "implementer": {
                        "model": "gpt-5.6-sol",
                        "effort": "max",
                        "serviceTier": "default",
                    },
                    "assistant": UNUSED_ROLE_RUNTIME,
                },
            )
            self.assertEqual(
                store.get("w_main", "conversation")["codex_sessions"],
                {"orchestrator": "sol-session"},
            )

            with self.assertRaisesRegex(ValueError, "family is locked"):
                store.update_codex_runtime(
                    "w_main",
                    "conversation",
                    role_runtimes={
                        "orchestrator": {"model": DEEPSEEK_MODEL, "effort": "max"},
                        "implementer": {"model": "gpt-5.6-sol", "effort": "max"},
                    },
                )

    def test_legacy_backend_columns_migrate_onto_models(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            database_path = registry.database_path("w_main")
            database_path.parent.mkdir(parents=True, exist_ok=True)
            _write_legacy_conversation_database(database_path)

            store = Store(registry)
            conversation = store.get("w_main", "legacy")

            self.assertEqual(
                conversation["codex_runtime"],
                {
                    "orchestrator": {
                        "model": DEEPSEEK_MODEL,
                        "effort": "max",
                        "serviceTier": "default",
                    },
                    "implementer": {
                        "model": "gpt-5.6-sol",
                        "effort": "xhigh",
                        "serviceTier": "default",
                    },
                    "assistant": UNUSED_ROLE_RUNTIME,
                },
            )
            # The old rollout is still reusable: its backend id became a family.
            self.assertEqual(
                store.sessions_for_prompt(
                    "w_main",
                    "legacy",
                    "contract",
                    runtimes={
                        "orchestrator": {"model": DEEPSEEK_MODEL, "effort": "max"},
                    },
                ),
                {"orchestrator": "legacy-session"},
            )
            # A pre-`agent_mode` conversation keeps the two-role behavior it was
            # created with; the new column must not reinterpret its history.
            self.assertEqual(conversation["agent_mode"], "orchestrated")

    def test_single_mode_round_trips_through_create_and_update(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            store = Store(registry)
            store.create(
                "w_main",
                "conversation",
                "workspace-write",
                agent_mode="single",
                role_runtimes={
                    "assistant": {"model": DEEPSEEK_MODEL, "effort": "high"},
                },
            )
            store.set_codex_session(
                "w_main",
                "conversation",
                "assistant",
                "assistant-session",
                family="deepseek",
            )
            store.add_message("w_main", "conversation", "user", "start")

            conversation = store.get("w_main", "conversation")
            self.assertEqual(conversation["agent_mode"], "single")
            self.assertEqual(
                conversation["codex_runtime"]["assistant"],
                {
                    "model": DEEPSEEK_MODEL,
                    "effort": "high",
                    "serviceTier": "default",
                },
            )
            self.assertEqual(
                conversation["codex_sessions"], {"assistant": "assistant-session"}
            )

            # The family lock follows the mode's own role, not the orchestrator.
            with self.assertRaisesRegex(ValueError, "family is locked"):
                store.update_codex_runtime(
                    "w_main",
                    "conversation",
                    role_runtimes={"assistant": {"model": "gpt-5.6-sol"}},
                )
            # An unused role may still cross families freely.
            store.update_codex_runtime(
                "w_main",
                "conversation",
                role_runtimes={
                    "assistant": {"model": DEEPSEEK_MODEL, "effort": "max"},
                    "orchestrator": {"model": "gpt-5.6-sol"},
                },
            )
            self.assertEqual(
                store.get("w_main", "conversation")["codex_runtime"]["orchestrator"][
                    "model"
                ],
                "gpt-5.6-sol",
            )

    def test_generated_names_use_pending_compare_and_set(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            descriptor = registry.create(
                "Prompt fallback",
                workspace_id="w_pending",
                naming_state="pending",
            )
            store = Store(registry)
            conversation = store.create(
                "w_pending",
                "conversation",
                "workspace-write",
                naming_state="pending",
            )
            store.add_message(
                "w_pending",
                "conversation",
                "user",
                "Find the best tensor parallel configuration",
            )

            self.assertEqual(descriptor["naming_state"], "pending")
            self.assertEqual(conversation["naming_state"], "pending")
            self.assertEqual(
                store.get("w_pending", "conversation")["title"],
                "Find the best tensor parallel configuration",
            )
            self.assertTrue(
                registry.apply_generated_name("w_pending", "Llama 3 H200 Study")
            )
            self.assertTrue(
                store.apply_generated_conversation_title(
                    "w_pending",
                    "conversation",
                    "Tensor Parallel Tradeoffs",
                )
            )
            self.assertFalse(
                registry.apply_generated_name("w_pending", "Late Workspace Name")
            )
            self.assertFalse(
                store.apply_generated_conversation_title(
                    "w_pending",
                    "conversation",
                    "Late Conversation Name",
                )
            )
            self.assertEqual(registry.get("w_pending")["naming_state"], "generated")
            self.assertEqual(
                store.get("w_pending", "conversation")["naming_state"], "generated"
            )

    def test_manual_workspace_rename_wins_over_generation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            registry.create(
                "Fallback",
                workspace_id="w_pending",
                naming_state="pending",
            )

            renamed = registry.update("w_pending", display_name="Operator name")

            self.assertEqual(renamed["naming_state"], "manual")
            self.assertFalse(registry.apply_generated_name("w_pending", "Late name"))
            self.assertEqual(registry.get("w_pending")["display_name"], "Operator name")

    def test_existing_database_migrates_conversations_to_manual(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            database_path = registry.database_path("w_main")
            connection = sqlite3.connect(database_path)
            connection.executescript(
                """
                CREATE TABLE schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                );
                INSERT INTO schema_migrations(version, applied_at) VALUES (1, 0);
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    sandbox TEXT NOT NULL,
                    autonomous INTEGER NOT NULL,
                    prompt_fingerprint TEXT,
                    peer_workspace TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                INSERT INTO conversations
                    VALUES ('legacy', 'Legacy title', 'read-only', 0, NULL, NULL, 1, 2);
                """
            )
            connection.commit()
            connection.close()

            store = Store(registry)
            legacy = store.get("w_main", "legacy")

            assert legacy is not None
            self.assertEqual(legacy["naming_state"], "manual")

    def test_global_compatibility_index_keeps_workspace_identity(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            registry.create("Second", workspace_id="w_second")
            store = Store(registry)
            store.create(
                "w_main",
                "main-conversation",
                "read-only",
                title="Main chat",
                updated_at=10,
            )
            store.create(
                "w_second",
                "second-conversation",
                "workspace-write",
                title="Second chat",
                updated_at=20,
            )

            conversations = store.list_all()

            self.assertEqual(
                [
                    (conversation["workspace_id"], conversation["id"])
                    for conversation in conversations
                ],
                [
                    ("w_second", "second-conversation"),
                    ("w_main", "main-conversation"),
                ],
            )

    def test_registry_uses_stable_workspace_ids_not_root_order(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            registry.create("Second", workspace_id="w_second")
            registry.create("Third", workspace_id="w_third")
            payload = json.loads(registry.registry_path.read_text("utf-8"))

            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(
                {row["workspace_id"] for row in payload["workspaces"]},
                {"w_main", "w_second", "w_third"},
            )
            self.assertTrue(
                all(
                    not Path(row["logs_root"]).is_absolute()
                    for row in payload["workspaces"]
                )
            )

    def test_turn_events_preserve_managed_callback_order(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            registry = self.make_registry(temporary_directory)
            store = Store(registry)
            store.create("w_main", "conversation", "workspace-write")
            store.start_turn("w_main", "conversation", "turn")
            store.append_turn_event(
                "w_main",
                "turn",
                "simulation.requested",
                {"experimentId": "e_test", "status": "requested"},
            )
            store.append_turn_event(
                "w_main",
                "turn",
                "tool_call",
                {"text": "not persisted in the rendered story"},
            )
            store.append_turn_event(
                "w_main",
                "turn",
                "experiment.ready",
                {"experimentId": "e_test", "status": "ready"},
            )

            events = store.list_turn_events(
                "w_main",
                "turn",
                kinds={"simulation.requested", "experiment.ready"},
            )

            self.assertEqual(
                [event["kind"] for event in events],
                ["simulation.requested", "experiment.ready"],
            )
            self.assertEqual([event["sequence"] for event in events], [0, 2])


class LegacyMigrationTest(unittest.TestCase):
    def test_moves_validates_imports_and_archives_legacy_state(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            main_dir = root / "main"
            (main_dir / "logs").mkdir(parents=True)
            registry = WorkspaceRegistry(root / "agent-workspaces", main_dir=main_dir)
            legacy_workspaces = root / "legacy-workspaces"
            legacy_conversation_dir = legacy_workspaces / "abc123"
            (legacy_conversation_dir / "main" / "logs").mkdir(parents=True)
            (legacy_conversation_dir / "main" / "tracked.txt").write_text("preserve me")
            (legacy_conversation_dir / "codex-home" / "sessions").mkdir(parents=True)
            (
                legacy_conversation_dir / "codex-home" / "sessions" / "one.jsonl"
            ).write_text("{}\n")
            conversations_path = root / "conversations.json"
            conversations_path.write_text(
                json.dumps(
                    {
                        "conversations": [
                            {
                                "id": "abc123",
                                "title": "Legacy",
                                "sandbox": "workspace-write",
                                "messages": [
                                    {"role": "user", "content": "hello", "ts": 1},
                                    {"role": "assistant", "content": "hi", "ts": 2},
                                ],
                                "codex_sessions": {"orchestrator": "session-1"},
                                "created_at": 1,
                                "updated_at": 2.5,
                            }
                        ]
                    }
                )
            )

            result = execute_migration(
                registry=registry,
                conversations_path=conversations_path,
                legacy_workspaces_dir=legacy_workspaces,
            )

            self.assertEqual(result["imported_conversations"], 1)
            imported = Store(registry).get("w_legacy_abc123", "abc123")
            assert imported is not None
            self.assertEqual(
                [message["content"] for message in imported["messages"]],
                ["hello", "hi"],
            )
            self.assertEqual(imported["codex_sessions"]["orchestrator"], "session-1")
            self.assertEqual(imported["created_at"], 1)
            self.assertEqual(imported["updated_at"], 2.5)
            self.assertEqual(
                (registry.repo_path("w_legacy_abc123") / "tracked.txt").read_text(),
                "preserve me",
            )
            self.assertEqual(
                (
                    registry.codex_home("w_legacy_abc123", "abc123")
                    / "sessions"
                    / "one.jsonl"
                ).read_text(),
                "{}\n",
            )
            self.assertFalse(conversations_path.exists())
            self.assertFalse(legacy_workspaces.exists())
            archive_markers = list(
                (registry.root / "migrations").glob("*/migration.complete.json")
            )
            self.assertEqual(len(archive_markers), 1)

    def test_completed_timestamp_repair_is_exact_and_idempotent(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            main_dir = root / "main"
            (main_dir / "logs").mkdir(parents=True)
            registry = WorkspaceRegistry(root / "agent-workspaces", main_dir=main_dir)
            store = Store(registry)
            store.create(
                "w_main",
                "legacy",
                "workspace-write",
                created_at=10,
                updated_at=10,
            )
            store.add_message("w_main", "legacy", "user", "hello", ts=11)
            archive_root = registry.root / "migrations" / "completed"
            legacy_source = archive_root / "legacy-source"
            legacy_source.mkdir(parents=True)
            (legacy_source / "conversations.json").write_text(
                json.dumps(
                    {
                        "conversations": [
                            {
                                "id": "legacy",
                                "title": "hello",
                                "sandbox": "workspace-write",
                                "messages": [
                                    {"role": "user", "content": "hello", "ts": 11}
                                ],
                                "created_at": 10,
                                "updated_at": 11.25,
                            }
                        ]
                    }
                )
            )

            first = repair_completed_timestamps(
                registry=registry,
                archive_root=archive_root,
            )
            second = repair_completed_timestamps(
                registry=registry,
                archive_root=archive_root,
            )
            repaired = store.get("w_main", "legacy")

            assert repaired is not None
            self.assertEqual(repaired["created_at"], 10)
            self.assertEqual(repaired["updated_at"], 11.25)
            self.assertEqual(first, second)
            self.assertEqual(first["repaired"], 1)


if __name__ == "__main__":
    unittest.main()
