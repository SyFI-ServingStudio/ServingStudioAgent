import asyncio
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

        tool_calls = [event["text"] for event in events if event["kind"] == "tool_call"]

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


class InterruptGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_blind_window_opens_at_the_start_and_closes_exactly_once(
        self,
    ) -> None:
        """`role_start`/`role_ready` bracket the window an interrupt must not fall in.

        Until a role has produced something, the prompt carrying the handoff — a
        delegated task, or an implementer summary on its way back — is not yet
        durable in its rollout, so cancelling there would drop it. A call whose
        only output is its final answer is the case worth pinning: nothing
        reaches the client until the very end, and the window still has to close
        or a client waiting on it would wait forever.
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
                    label="implementer",
                    workspace_id="w_test",
                    conversation_id="conversation-1",
                    turn_id="turn-1",
                    model_id="gpt-5.6-sol",
                    effort="xhigh",
                )
            ]

        kinds = [event["kind"] for event in events]
        self.assertEqual(kinds[0], "role_start")
        self.assertEqual(events[0]["role"], "implementer")
        self.assertEqual(kinds.count("role_ready"), 1)
        self.assertEqual(events[kinds.index("role_ready")]["role"], "implementer")
        # The session event is Codex answering the CLI, not the model, and the
        # ticking waiting line is this process talking to itself; neither counts
        # as the role having spoken.
        self.assertLess(kinds.index("session"), kinds.index("role_ready"))
        self.assertLess(kinds.index("role_ready"), kinds.index("final"))


if __name__ == "__main__":
    unittest.main()


# Announces its pid and then waits: enough to tell whether a cancellation that
# lands during startup still reaches the process that was just spawned.
FAKE_CODEX_SLOW_START = """
import sys, time
print("started", flush=True)
sys.stdin.read()
time.sleep(60)
"""


class HandoffBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Where a role's blind window begins, and what a Stop inside it reaches."""

    async def test_says_the_role_has_started_before_anything_can_suspend(self) -> None:
        """`role_start` is the only thing that says "not safe to stop here".

        Starting the process and writing the prompt both suspend, and the second
        of them *is* the handoff. Announcing the role after them left both inside
        the previous role's ready window, where a Stop is authorised at once —
        so the handoff would be cancelled away and the next turn would resume a
        role that never received its instructions.
        """
        spawned = asyncio.Event()
        real_exec = asyncio.create_subprocess_exec

        async def watched(*args: object, **kwargs: object):
            spawned.set()
            return await real_exec(*args, **kwargs)

        events: list[dict] = []
        with (
            patch.object(
                codex_cli,
                "build_codex_exec_command",
                return_value=[sys.executable, "-c", FAKE_CODEX_SLOW_START],
            ),
            patch.object(asyncio, "create_subprocess_exec", watched),
        ):
            stream = run_codex(
                "container",
                "question",
                label="implementer",
                workspace_id="w_test",
                conversation_id="conversation-1",
                turn_id="turn-1",
                model_id="gpt-5.6-sol",
                effort="xhigh",
            )
            events.append(await stream.__anext__())
            self.assertFalse(spawned.is_set())
            await stream.aclose()

        self.assertEqual(events[0], {"kind": "role_start", "role": "implementer"})

    async def test_a_cancel_during_startup_still_stops_the_process(self) -> None:
        """The window between spawning and streaming is not unattended.

        Cancelling there used to miss the cleanup entirely: the container
        process had been created but the `try` that stops it had not been
        entered, so it outlived the turn that started it.
        """
        signalled: list[str] = []
        spawned: list[object] = []
        real_exec = asyncio.create_subprocess_exec

        async def watched(*args: object, **kwargs: object):
            process = await real_exec(*args, **kwargs)
            spawned.append(process)
            return process

        async def held_prompt(self: object, process: object) -> None:
            # Stands in for `stdin.drain()` blocking on backpressure, which is
            # what makes this window wide enough to hit.
            await asyncio.sleep(60)

        with (
            patch.object(
                codex_cli,
                "build_codex_exec_command",
                return_value=[sys.executable, "-c", FAKE_CODEX_SLOW_START],
            ),
            patch.object(asyncio, "create_subprocess_exec", watched),
            patch.object(codex_cli.CodexExecCall, "_write_prompt", held_prompt),
            patch.object(
                codex_cli.CodexExecCall,
                "_signal_container_codex",
                lambda self, signal: signalled.append(signal) or _nothing(),
            ),
        ):
            stream = run_codex(
                "container",
                "question",
                label="implementer",
                workspace_id="w_test",
                conversation_id="conversation-1",
                turn_id="turn-1",
                model_id="gpt-5.6-sol",
                effort="xhigh",
            )
            await stream.__anext__()  # role_start
            await stream.__anext__()  # the "starting Codex" line

            consuming = asyncio.create_task(_drain(stream))
            for _ in range(200):
                if spawned:
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(spawned, "the process was never started")

            consuming.cancel()
            try:
                await consuming
            except asyncio.CancelledError:
                pass

        self.assertEqual(signalled[:1], ["INT"])
        process = spawned[0]
        await asyncio.wait_for(process.wait(), timeout=5)


class AwaitStoppedTests(unittest.IsolatedAsyncioTestCase):
    """Waiting for a helper task must not absorb a Stop aimed at the turn.

    Both callers wait in a `finally`, which is where a Stop that arrives at the
    end of a role lands. If that cancellation is swallowed there, the turn walks
    on into its next role while the conversation has already recorded the Stop
    as delivered — and from then on the turn cannot be stopped at all.
    """

    async def test_a_helper_that_was_cancelled_does_not_stop_its_waiter(self) -> None:
        helper = asyncio.create_task(asyncio.sleep(3600))
        helper.cancel()

        await codex_cli._await_stopped(helper)

        self.assertTrue(helper.cancelled())

    async def test_a_helper_that_failed_does_not_stop_its_waiter(self) -> None:
        # Nothing reads a drained stderr task's result, and its failure says
        # nothing about whether the role produced an answer.
        async def fails() -> None:
            raise RuntimeError("the stderr reader broke")

        await codex_cli._await_stopped(asyncio.create_task(fails()))

    async def test_being_cancelled_while_waiting_stops_the_waiter(self) -> None:
        # A helper that ignores its own cancellation holds the wait open, which
        # is the window a Stop has to land in.
        async def stubborn() -> None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(3600)

        helper = asyncio.create_task(stubborn())
        await asyncio.sleep(0)
        helper.cancel()

        waiting = asyncio.create_task(codex_cli._await_stopped(helper))
        await asyncio.sleep(0.05)
        waiting.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertTrue(waiting.cancelled(), "the Stop was absorbed by the wait")
        helper.cancel()


async def _nothing() -> None:
    return None


async def _drain(stream) -> None:
    async for _ in stream:
        pass
