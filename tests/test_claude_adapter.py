"""Claude subprocess lifecycle without Docker or live provider calls."""

import asyncio
import json
import logging
import os
import signal
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vibesim_agent.domain.roles import Role
from vibesim_agent.providers.base import AgentRequest, Model, OutputMode, Selection
from vibesim_agent.providers.claude import adapter as adapter_module
from vibesim_agent.runtime import execution as execution_module
from vibesim_agent.providers.claude.adapter import ClaudeAdapter
from vibesim_agent.runtime.invocation import InvocationHome


class LocalCommand:
    def __init__(self, code, schema=None):
        self.code = code
        self.schema = schema
        self.calls = []

    def output_schema(self, request):
        return self.schema if request.structured_output else None

    def build_tracked(self, request, *, home, pid_file):
        self.calls.append((request, home))
        return [
            sys.executable,
            "-u",
            "-c",
            "import os,sys,pathlib; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
            + self.code,
            pid_file,
        ]


from tests.test_codex_adapter import LocalExecution


class ClaudeAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.home = Path(self.enterContext(TemporaryDirectory()))
        self.execution = LocalExecution()
        self.model = Model("claude", "Claude", ("high",), "high")
        self.request = AgentRequest(
            "w",
            "c",
            "t",
            Role.ASSISTANT,
            "question",
            self.execution,
            Selection("profile", self.model, "high", "default", "claude:v1"),
        )

    def adapter(self, code, *, timeout=2, schema=None):
        return ClaudeAdapter(
            LocalCommand(code, schema),
            home=lambda _: InvocationHome(self.home, str(self.home)),
            idle_timeout=timeout,
            logger=logging.getLogger("claude-adapter-test"),
            poll_interval=0.02,
        )

    async def consume(self, adapter, request=None):
        return [event async for event in adapter.run(request or self.request)]

    async def test_first_resume_session_progress_usage_final_and_private_home(self):
        records = [
            {"type": "system", "subtype": "init", "session_id": "saved"},
            {
                "type": "assistant",
                "uuid": "a",
                "message": {
                    "content": [
                        {"type": "text", "text": "progress"},
                        {"type": "tool_use", "name": "Read"},
                    ]
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "result": "answer",
                "usage": {
                    "input_tokens": 3,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 7,
                    "output_tokens": 2,
                },
            },
        ]
        adapter = self.adapter(
            "assert sys.stdin.read() == 'question'; print("
            + repr("\n".join(json.dumps(r) for r in records))
            + ")"
        )
        for session in (None, "saved"):
            events = await self.consume(
                adapter,
                replace(
                    self.request,
                    session_id=session,
                    execution_id="call-" + str(session),
                ),
            )
            self.assertEqual(events[0], {"kind": "role_start", "role": "assistant"})
            self.assertEqual(events[-1], {"kind": "final", "text": "answer"})
            self.assertEqual(sum(e["kind"] == "role_ready" for e in events), 1)
            self.assertEqual(
                next(e for e in events if e["kind"] == "usage")["tokens"],
                {"read": 7, "prefill": 8, "output": 2},
            )
            self.assertTrue(any(e["kind"] == "intermediate_output" for e in events))
            self.assertTrue(any(e["kind"] == "tool_call" for e in events))
        self.assertEqual(
            [call[0].session_id for call in adapter.command.calls], [None, "saved"]
        )
        self.assertEqual(
            [call[1] for call in adapter.command.calls], [str(self.home)] * 2
        )
        self.assertEqual(list(self.home.iterdir()), [])

    async def test_final_without_newline_and_structured_schema_validation(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        for value, valid in (({"answer": "yes"}, True), ({"answer": 3}, False)):
            result = {
                "type": "result",
                "subtype": "success",
                "structured_output": value,
            }
            adapter = self.adapter(
                "sys.stdout.write(" + repr(json.dumps(result)) + ")", schema=schema
            )
            request = replace(
                self.request, output_schema=Path("/contracts/assistant.schema.json")
            )
            events = await self.consume(adapter, request)
            self.assertEqual(sum(e["kind"] == "role_ready" for e in events), 1)
            if valid:
                self.assertEqual(json.loads(events[-1]["text"]), value)
            else:
                self.assertEqual(events[-1]["failure"]["code"], "agent_invalid_output")

    async def test_prompt_only_model_ignores_request_schema(self):
        request = replace(
            self.request,
            output_schema=Path("/contracts/assistant.schema.json"),
            selection=replace(
                self.request.selection,
                model=replace(self.model, output_mode=OutputMode.PROMPT),
            ),
        )
        adapter = self.adapter(
            "print("
            + repr(
                json.dumps(
                    {"type": "result", "subtype": "success", "result": "plain answer"}
                )
            )
            + ")"
        )
        self.assertEqual(
            (await self.consume(adapter, request))[-1],
            {"kind": "final", "text": "plain answer"},
        )

    async def test_nonzero_exit_is_failure_and_stderr_never_becomes_answer(self):
        code = (
            "sys.stderr.write('secret' * 20000); print("
            + repr(
                json.dumps({"type": "result", "subtype": "success", "result": "answer"})
            )
            + "); sys.exit(2)"
        )
        events = await self.consume(self.adapter(code))
        self.assertEqual(
            events[-1],
            {
                "kind": "final",
                "text": "",
                "failure": {"code": "agent_runtime_failure", "status": 0},
            },
        )
        self.assertNotIn("secret", json.dumps(events))

    async def test_idle_timeout_stops_unready_child_and_keeps_marker(self):
        async with asyncio.timeout(3):
            events = await self.consume(
                self.adapter("import time; time.sleep(30)", timeout=0.07)
            )
        self.assertFalse(any(e["kind"] == "role_ready" for e in events))
        self.assertEqual(events[-1]["failure"]["code"], "agent_call_timeout")
        self.assertFalse(list(self.home.glob("*.pid")))
        self.assertTrue(
            (self.home / f"call-{self.request.execution_id}.pid.cancel").exists()
        )

    async def test_signal_helper_failure_still_reaches_kill(self):
        adapter = self.adapter("import time; time.sleep(30)", timeout=0.07)
        original = self.execution.local_signal
        signals = []

        async def interrupted_helper(container, pid_file, name):
            signals.append(name)
            if name != "KILL":
                raise TimeoutError("helper blocked")
            await original(container, pid_file, name)

        self.execution.signal = interrupted_helper
        with patch.object(execution_module, "SIGNAL_GRACE", 0.02):
            async with asyncio.timeout(3):
                events = await self.consume(adapter)
        self.assertEqual(signals, ["INT", "TERM", "KILL"])
        self.assertEqual(events[-1]["failure"]["code"], "agent_call_timeout")

    async def test_cancelled_consumer_stops_process_even_when_marker_write_fails(self):
        adapter = self.adapter("import time; time.sleep(30)")
        task = asyncio.create_task(self.consume(adapter))
        async with asyncio.timeout(3):
            while not list(self.home.glob("*.pid")):
                await asyncio.sleep(0.01)
            with patch.object(Path, "touch", side_effect=OSError("disk full")) as touch:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                touch.assert_called_once()
        self.assertFalse(list(self.home.glob("*.pid")))
