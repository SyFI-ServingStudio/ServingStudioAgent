import sys
import unittest
from unittest.mock import patch

from backend.codex_runtime import codex_cli
from backend.codex_runtime.codex_cli import run_codex

# Stands in for `docker exec ... codex exec`: announces the thread, stays quiet
# for a while (the model's time to first token), then answers.
FAKE_CODEX = """
import json, sys, time
sys.stdin.read()
print(json.dumps({"type": "thread.started", "thread_id": "thread-1"}), flush=True)
time.sleep(1.6)  # the stdout poll wakes once a second, so out-tick the poll
print(json.dumps({
    "type": "item.completed",
    "item": {"type": "agent_message", "text": "done", "phase": ""},
}), flush=True)
"""


class FirstOutputWaitTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_wait_for_the_first_output_narrates_itself(self) -> None:
        """Codex is silent until its first item, so the caller must fill the gap.

        Without these the UI keeps showing whatever line preceded the call —
        "checking Docker Codex container..." — across CLI startup, session load
        and the whole time to first token.
        """
        with (
            patch.object(codex_cli, "FIRST_OUTPUT_TICK_SECONDS", 0.1),
            patch.object(
                codex_cli,
                "build_codex_exec_command",
                return_value=[sys.executable, "-c", FAKE_CODEX],
            ),
        ):
            events = [
                event
                async for event in run_codex(
                    "container",
                    "question",
                    label="orchestrator",
                    workspace_id="w_test",
                    conversation_id="conversation-1",
                    turn_id="turn-1",
                    model_id="gpt-5.6-sol",
                    effort="xhigh",
                )
            ]

        tool_calls = [
            event["text"] for event in events if event["kind"] == "tool_call"
        ]

        # Immediately, before Codex has said anything at all.
        self.assertEqual(
            tool_calls[0],
            "orchestrator: starting Codex with gpt-5.6-sol · xhigh...",
        )
        # After `thread.started` the remaining wait is time to first token, and
        # it keeps ticking so the line reads as alive rather than stuck.
        waiting = [line for line in tool_calls if "waiting for first output" in line]
        self.assertTrue(waiting, tool_calls)
        self.assertTrue(
            all(
                line.startswith("orchestrator: waiting for first output from")
                for line in waiting
            ),
            waiting,
        )
        # ...and it stops once the model actually answers.
        self.assertEqual(tool_calls, tool_calls[: len(waiting) + 1])
        self.assertEqual(events[-1], {"kind": "final", "text": "done"})


# The real shape of the outage: Codex exhausts its own reconnects, writes the
# gateway status to stdout as an error event, and then exits 0 — so nothing at
# the process level says the call failed.
FAKE_CODEX_503 = """
import json, sys
sys.stdin.read()
print(json.dumps({"type": "thread.started", "thread_id": "thread-1"}), flush=True)
for attempt in range(1, 6):
    print(json.dumps({
        "type": "error",
        "message": (
            f"Reconnecting... {attempt}/5 (unexpected status 503 Service "
            "Unavailable: Service temporarily unavailable, url: "
            "http://cayenne.cs.washington.edu:3456/responses, request id: abc)"
        ),
    }), flush=True)
print(json.dumps({
    "type": "error",
    "message": (
        "unexpected status 503 Service Unavailable: Service temporarily "
        "unavailable, url: http://cayenne.cs.washington.edu:3456/responses, "
        "request id: def"
    ),
}), flush=True)
sys.exit(0)
"""


class UpstreamFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_outage_produces_a_failed_final(self) -> None:
        """End to end through a real subprocess: the CLI exits 0 with no answer,
        so the failure has to be read out of the stream itself."""
        with patch.object(
            codex_cli,
            "build_codex_exec_command",
            return_value=[sys.executable, "-c", FAKE_CODEX_503],
        ):
            events = [
                event
                async for event in run_codex(
                    "container",
                    "question",
                    label="orchestrator",
                    workspace_id="w_test",
                    conversation_id="conversation-1",
                    turn_id="turn-1",
                    model_id="gpt-5.6-sol",
                    effort="xhigh",
                )
            ]

        # The reconnect attempts stay visible as ordinary advisory lines.
        warnings = [
            event["text"]
            for event in events
            if event["kind"] == "tool_call" and "Reconnecting" in event["text"]
        ]
        self.assertEqual(len(warnings), 5)

        self.assertEqual(events[-1]["kind"], "final")
        self.assertEqual(
            events[-1]["failure"], {"code": "upstream_unavailable", "status": 503}
        )
        self.assertNotIn("cayenne", events[-1]["text"])


if __name__ == "__main__":
    unittest.main()
