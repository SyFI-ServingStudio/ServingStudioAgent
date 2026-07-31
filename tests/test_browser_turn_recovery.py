from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend import app as app_module
from backend.store import Store, WorkspaceRegistry


class TurnFailureTests(unittest.TestCase):
    def test_backend_restart_recovers_persisted_activity_once(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            main_dir = Path(temporary_directory) / "main"
            (main_dir / "logs").mkdir(parents=True)
            registry = WorkspaceRegistry(
                Path(temporary_directory) / "agent-workspaces",
                main_dir=main_dir,
            )
            store = Store(registry)
            store.create("w_main", "conversation", "workspace-write")
            store.add_message("w_main", "conversation", "user", "continue")
            store.start_turn("w_main", "conversation", "turn")
            store.append_turn_event(
                "w_main",
                "turn",
                "intermediate_output",
                {"kind": "intermediate_output", "role": "orchestrator", "text": "Plan"},
            )
            store.append_turn_event(
                "w_main",
                "turn",
                "decision",
                {"kind": "decision", "action": "run_implementer", "task": "Implement"},
            )

            with (
                patch.object(app_module, "store", store),
                patch.object(app_module, "remove_managed_context"),
            ):
                self.assertEqual(app_module._recover_orphaned_browser_turns(), 1)
                self.assertEqual(app_module._recover_orphaned_browser_turns(), 0)

            conversation = store.get("w_main", "conversation")
            self.assertIsNotNone(conversation)
            assistant = conversation["messages"][-1]
            self.assertEqual(assistant["role"], "assistant")
            self.assertIn("backend restarted", assistant["content"])
            self.assertEqual(
                [event["kind"] for event in assistant["activity"]],
                ["intermediate_output", "decision", "error"],
            )
            self.assertEqual(store.list_running_turns("w_main"), [])

    def test_storage_failure_is_user_safe_and_actionable(self) -> None:
        failure = app_module._turn_failure(
            RuntimeError(
                "Command ['docker', 'run', '-d', '--mount', '/host'] failed: "
                "no space left on device"
            )
        )

        self.assertEqual(failure["code"], "runtime_storage_full")
        self.assertIn("host disk is full", failure["message"])
        self.assertNotIn("docker", failure["message"].lower())
        self.assertNotIn("/host", failure["message"])

    def test_unknown_failure_hides_internal_diagnostics(self) -> None:
        failure = app_module._turn_failure(
            RuntimeError("secret subprocess command and host path")
        )

        self.assertEqual(failure["code"], "agent_runtime_failure")
        self.assertNotIn("secret", failure["message"])

    @patch.object(app_module.store, "list_turn_events")
    def test_managed_run_callbacks_project_to_reloadable_job_cards(
        self,
        list_turn_events,
    ) -> None:
        list_turn_events.return_value = [
            {
                "sequence": 4,
                "kind": "experiment.ready",
                "payload": {
                    "workspaceId": "w_test",
                    "experimentId": "e_test",
                    "experimentPath": "20260728_test",
                    "jobId": "j_test",
                    "status": "ready",
                },
            }
        ]

        activity = app_module._managed_turn_activity("w_test", "turn")

        self.assertEqual(
            activity,
            [
                {
                    "kind": "job",
                    "workspaceId": "w_test",
                    "experimentId": "e_test",
                    "experimentPath": "20260728_test",
                    "jobId": "j_test",
                    "status": "ready",
                }
            ],
        )


class ResumeTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_conversation_returns_no_content(self) -> None:
        conversation_id = "idle-conversation"
        app_module._active_browser_turns.pop(("w_main", conversation_id), None)

        with patch.object(
            app_module.store,
            "get",
            return_value={"id": conversation_id},
        ):
            response = await app_module.resume_message_stream(
                "w_main",
                conversation_id,
            )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.body, b"")

    async def test_browser_done_reports_background_naming_schedule(self) -> None:
        async def fake_run_turn(*args, **kwargs):
            yield {"kind": "final", "text": "The answer"}

        with TemporaryDirectory() as temporary_directory:
            main_dir = Path(temporary_directory) / "main"
            (main_dir / "logs").mkdir(parents=True)
            registry = WorkspaceRegistry(
                Path(temporary_directory) / "agent-workspaces",
                main_dir=main_dir,
            )
            store = Store(registry)
            store.create(
                "w_main",
                "conversation",
                "workspace-write",
                naming_state="pending",
            )
            store.start_turn("w_main", "conversation", "turn")
            active_turn = app_module.ActiveBrowserTurn(turn_id="turn")
            with (
                patch.object(app_module, "store", store),
                patch.object(app_module, "run_turn", fake_run_turn),
                patch.object(app_module, "schedule_auto_naming", return_value=True),
                patch.object(app_module, "remove_managed_context"),
            ):
                await app_module._run_browser_turn(
                    workspace_id="w_main",
                    cid="conversation",
                    text="The question",
                    sandbox="workspace-write",
                    sessions={},
                    turn_id="turn",
                    prompt_fingerprint="fingerprint",
                    autonomous=False,
                    analyzer_context=None,
                    active_turn=active_turn,
                )

            done_payload = next(
                json.loads(event.split("data: ", 1)[1])
                for event in active_turn.events
                if event.startswith("event: done")
            )
            self.assertTrue(done_payload["naming_scheduled"])

    async def test_agent_result_reports_background_naming_schedule(self) -> None:
        async def fake_run_turn(*args, **kwargs):
            yield {"kind": "final", "text": "The answer"}

        with TemporaryDirectory() as temporary_directory:
            main_dir = Path(temporary_directory) / "main"
            (main_dir / "logs").mkdir(parents=True)
            registry = WorkspaceRegistry(
                Path(temporary_directory) / "agent-workspaces",
                main_dir=main_dir,
            )
            store = Store(registry)
            store.create(
                "w_main",
                "conversation",
                "workspace-write",
                naming_state="pending",
            )
            with (
                patch.object(app_module, "store", store),
                patch.object(app_module, "run_turn", fake_run_turn),
                patch.object(app_module, "schedule_auto_naming", return_value=True),
                patch.object(app_module, "remove_managed_context"),
            ):
                result = await app_module.agent_send_message(
                    "w_main",
                    "conversation",
                    app_module.AgentSendMessage(text="The question"),
                    None,
                )

            self.assertTrue(result["ok"])
            self.assertTrue(result["naming_scheduled"])


if __name__ == "__main__":
    unittest.main()
