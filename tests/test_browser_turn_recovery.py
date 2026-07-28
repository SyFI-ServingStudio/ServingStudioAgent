from __future__ import annotations

import unittest
from unittest.mock import patch

from backend import app as app_module


class TurnFailureTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
