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


class ResumeTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_conversation_returns_no_content(self) -> None:
        conversation_id = "idle-conversation"
        app_module._active_browser_turns.pop(conversation_id, None)

        with patch.object(
            app_module.store,
            "get",
            return_value={"id": conversation_id},
        ):
            response = await app_module.resume_message_stream(conversation_id)

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.body, b"")


if __name__ == "__main__":
    unittest.main()
