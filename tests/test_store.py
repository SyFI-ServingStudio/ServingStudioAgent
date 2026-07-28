import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.migrate_workspaces import execute_migration
from backend.store import Store, WorkspaceRegistry


class WorkspaceStoreTest(unittest.TestCase):
    def make_registry(self, temporary_directory: str) -> WorkspaceRegistry:
        root = Path(temporary_directory)
        main_dir = root / "main"
        (main_dir / "logs").mkdir(parents=True)
        return WorkspaceRegistry(root / "agent-workspaces", main_dir=main_dir)

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
                all(not Path(row["logs_root"]).is_absolute() for row in payload["workspaces"])
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
                "progress",
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
            (legacy_conversation_dir / "codex-home" / "sessions" / "one.jsonl").write_text(
                "{}\n"
            )
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
                                "updated_at": 2,
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
            self.assertEqual([message["content"] for message in imported["messages"]], ["hello", "hi"])
            self.assertEqual(imported["codex_sessions"]["orchestrator"], "session-1")
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


if __name__ == "__main__":
    unittest.main()
