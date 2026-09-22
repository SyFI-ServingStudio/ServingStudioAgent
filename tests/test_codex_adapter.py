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
from vibesim_agent.providers.base import AgentRequest, Model, Selection
from vibesim_agent.providers.codex.adapter import CodexAdapter, CodexHome
from vibesim_agent.providers.codex import adapter as adapter_module
from vibesim_agent.providers.codex.command import CodexCommand
from tests.runtime_fixtures import execution_environment
from vibesim_agent.runtime import execution as execution_module
from vibesim_agent.runtime.invocation import RemoteInvocation, signal_remote
from vibesim_agent.runtime.process import kill_process


class LocalCommand:
    def __init__(self, code):
        self.code = code
        self.requests = []

    def build_tracked(self, request, *, home, pid_file):
        self.requests.append(request)
        return [
            sys.executable,
            "-u",
            "-c",
            "import os,sys,pathlib; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
            + self.code,
            pid_file,
        ]


class LocalExecution:
    """Stands in for the transport: a real local child, signalled by pid file.

    `signal` is reassignable so a test can inject an unresponsive or failing
    helper, which is what the Docker transport's failure modes look like from
    the adapter's side.
    """

    cwd = None
    start_new_session = False
    kill = staticmethod(kill_process)
    agent_prompt = "/workspace/AGENTS.md"
    managed_context = "/managed/context.json"
    stop_timeout = execution_module.STOP_TIMEOUT

    def __init__(self):
        self.signal = self.local_signal

    async def local_signal(self, container, pid_file, name):
        path = Path(pid_file)
        if path.exists():
            try:
                os.kill(int(path.read_text()), getattr(signal, "SIG" + name))
            except ProcessLookupError:
                pass

    def command(self, arguments, *, environment, inherited=(), pid_file, label):
        return list(arguments)

    def spawn_environment(self, process_environment):
        return dict(process_environment) if process_environment is not None else None

    def home_path(self, home):
        return home.container

    def invocation(self, *, execution_id, home, logger, label, process_environment):
        return RemoteInvocation(
            execution_id=execution_id,
            container="local",
            home=home,
            signal=lambda *arguments: self.signal(*arguments),
            logger=logger,
            label=label,
            signal_grace=execution_module.SIGNAL_GRACE,
        )


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.execution = LocalExecution()

    async def test_signal_helper_timeout_is_reaped(self):
        original = asyncio.create_subprocess_exec
        helpers = []

        async def sleeping_helper(*args, **kwargs):
            process = await original(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
            helpers.append(process)
            return process

        with TemporaryDirectory() as directory:
            adapter = self.adapter(directory, "pass")
            with patch("asyncio.create_subprocess_exec", sleeping_helper), patch.object(execution_module, "SIGNAL_TIMEOUT", 0.05):
                with self.assertRaises(TimeoutError):
                    await signal_remote(
                        "container", "pid", "INT",
                        timeout=execution_module.SIGNAL_TIMEOUT,
                        label="Codex", environment=None,
                    )
            self.assertIsNotNone(helpers[0].returncode)

    async def test_signal_timeout_continues_escalation_with_sufficient_budget(self):
        with TemporaryDirectory() as directory:
            adapter = self.adapter(directory, "import time; time.sleep(30)", timeout=0.05)
            original = self.execution.local_signal
            signals = []

            async def signal_with_timeout(container, pid_file, name):
                signals.append(name)
                if name in {"INT", "TERM"}:
                    raise TimeoutError("unresponsive signal helper")
                await original(container, pid_file, name)

            self.execution.signal = signal_with_timeout
            self.assertEqual(execution_module.SIGNAL_GRACE, 5)
            self.assertGreaterEqual(execution_module.STOP_TIMEOUT,
                                    3 * (2 * execution_module.SIGNAL_TIMEOUT + execution_module.SIGNAL_GRACE))
            with patch.object(execution_module, "SIGNAL_GRACE", 0.03):
                async with asyncio.timeout(3):
                    events = [event async for event in adapter.run(self.request())]
            self.assertEqual(signals, ["INT", "TERM", "KILL"])
            self.assertEqual(events[-1]["failure"]["code"], "codex_call_timeout")

    async def test_marker_write_failure_still_stops_known_remote_pid(self):
        with TemporaryDirectory() as directory:
            adapter = self.adapter(directory, "import time; time.sleep(30)")
            original = self.execution.local_signal
            signals = []

            async def record_signal(container, pid_file, name):
                signals.append(name)
                await original(container, pid_file, name)

            self.execution.signal = record_signal
            request = self.request()
            task = asyncio.create_task(self.consume(adapter, request))
            async with asyncio.timeout(3):
                while not list(Path(directory).glob("call-*.pid")):
                    await asyncio.sleep(0.01)
                with patch.object(Path, "touch", side_effect=OSError("disk full")) as touch:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    touch.assert_called_once()
            self.assertEqual(signals, ["INT"])
            self.assertFalse(list(Path(directory).glob("call-*.pid")))

    async def test_delayed_remote_shell_observes_persistent_startup_cancellation(self):
        class WrappedLocalExecution(LocalExecution):
            """Keeps the real pid wrapper, drops only the `docker exec` prefix.

            The wrapper is what this test is about: the remote shell can start
            after its local client has gone, and the `.cancel` check is the only
            thing that stops it.
            """

            def command(self, arguments, *, environment, inherited=(), pid_file, label):
                return ["sh", "-c", execution_module.PID_WRAPPER, label, pid_file, *arguments]

        class TrackedLocalCommand(CodexCommand):
            def arguments(self, request, *, home):
                return [sys.executable, "-c", "import sys; from pathlib import Path; Path(sys.argv[1]).touch()",
                        str(Path(home) / "executed")]

        self.execution = WrappedLocalExecution()

        class DelayedRemoteCommand:
            remote = None

            def build_tracked(self, request, *, home, pid_file):
                self.remote = TrackedLocalCommand(execution_environment()).build_tracked(
                    request, home=home, pid_file=pid_file
                )
                return [sys.executable, "-c", "import time; time.sleep(30)"]

        with TemporaryDirectory() as directory:
            command = DelayedRemoteCommand()
            adapter = CodexAdapter(command, home=lambda _: CodexHome(Path(directory), directory),
                                   idle_timeout=10, logger=logging.getLogger("adapter-test"))
            request = self.request()
            task = asyncio.create_task(self.consume(adapter, request))
            with patch.object(execution_module, "SIGNAL_GRACE", 0.03):
                async with asyncio.timeout(3):
                    while command.remote is None:
                        if task.done():
                            await task
                            self.fail("Adapter completed before constructing the remote command")
                        await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 3)
            marker = Path(directory) / f"call-{request.execution_id}.pid.cancel"
            self.assertTrue(marker.exists())
            # The remote side starts only after local cancellation/cleanup finished.
            remote = await asyncio.create_subprocess_exec(*command.remote)
            try:
                self.assertEqual(await asyncio.wait_for(remote.wait(), 3), 130)
            finally:
                if remote.returncode is None:
                    remote.kill()
                    await remote.wait()
            self.assertFalse((Path(directory) / "executed").exists())
            self.assertTrue(marker.exists())
            self.assertFalse(list(Path(directory).glob("call-*.pid")))
            fresh = self.request()
            fresh_pid = str(Path(directory) / f"call-{fresh.execution_id}.pid")
            followup = await asyncio.create_subprocess_exec(
                *TrackedLocalCommand(execution_environment()).build_tracked(
                    fresh, home=directory, pid_file=fresh_pid
                )
            )
            try:
                self.assertEqual(await asyncio.wait_for(followup.wait(), 3), 0)
            finally:
                if followup.returncode is None:
                    followup.kill()
                    await followup.wait()
            self.assertTrue((Path(directory) / "executed").exists())
            self.assertTrue(marker.exists())

    async def consume(self, adapter, request):
        return [event async for event in adapter.run(request)]

    async def test_rollout_recovers_final_missing_from_stdout(self):
        with TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "sessions").mkdir()
            record = json.dumps({"type": "event_msg", "payload": {
                "type": "task_complete", "last_agent_message": "durable answer",
            }})
            code = (
                "print(" + repr(json.dumps({"type": "thread.started", "thread_id": "saved"})) + "); "
                "pathlib.Path(" + repr(str(home / "sessions" / "saved.jsonl")) + ").write_text(" + repr(record) + ")"
            )
            events = [event async for event in self.adapter(directory, code).run(self.request())]
            self.assertEqual(events[-1], {"kind": "final", "text": "durable answer"})
            self.assertEqual(sum(e["kind"] == "role_ready" for e in events), 1)

    async def test_upstream_warning_is_not_readiness_evidence(self):
        with TemporaryDirectory() as directory:
            code = "print(" + repr(json.dumps({"type": "error", "message": "unexpected status 503 unavailable"})) + ")"
            events = [event async for event in self.adapter(directory, code).run(self.request())]
            self.assertFalse(any(e["kind"] == "role_ready" for e in events))
            self.assertEqual(events[-1]["failure"]["status"], 503)

    def request(self):
        model = Model("model", "Model", ("high",), "high")
        return AgentRequest(
            "w",
            "c",
            "t",
            Role.ASSISTANT,
            "question",
            self.execution,
            Selection("test", model, "high", "default", "scope"),
        )

    def adapter(self, directory, code, timeout=2):
        command = LocalCommand(code)
        return CodexAdapter(
            command,
            home=lambda _: CodexHome(Path(directory), directory),
            idle_timeout=timeout,
            logger=logging.getLogger("adapter-test"),
            poll_interval=0.02,
        )

    async def test_first_call_and_resume_emit_session_ready_usage_final(self):
        with TemporaryDirectory() as directory:
            records = [
                {"type": "thread.started", "thread_id": "saved"},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "answer",
                        "phase": "final_answer",
                    },
                },
            ]
            adapter = self.adapter(
                directory,
                "sys.stdin.read(); print("
                + repr("\n".join(json.dumps(r) for r in records))
                + ")",
            )
            request = self.request()
            for session in (None, "saved"):
                events = [
                    event
                    async for event in adapter.run(replace(request, session_id=session))
                ]
                self.assertEqual(events[0]["kind"], "role_start")
                self.assertEqual(events[-1], {"kind": "final", "text": "answer"})
                self.assertEqual(sum(e["kind"] == "role_ready" for e in events), 1)
                self.assertEqual(
                    next(e for e in events if e["kind"] == "session")["session_id"],
                    "saved",
                )
            self.assertEqual(
                [r.session_id for r in adapter.command.requests], [None, "saved"]
            )
            self.assertFalse(list(Path(directory).glob("call-*.pid")))

    async def test_idle_timeout_stops_process_and_removes_pid(self):
        with TemporaryDirectory() as directory:
            adapter = self.adapter(
                directory, "import time; time.sleep(30)", timeout=0.1
            )
            async with asyncio.timeout(5):
                events = [event async for event in adapter.run(self.request())]
            self.assertEqual(events[-1]["failure"]["code"], "codex_call_timeout")
            self.assertFalse(any(e["kind"] == "role_ready" for e in events))
            self.assertFalse(list(Path(directory).glob("call-*.pid")))

    async def test_cancel_after_spawn_cleans_execution(self):
        with TemporaryDirectory() as directory:
            adapter = self.adapter(directory, "import time; time.sleep(30)")

            async def consume():
                return [event async for event in adapter.run(self.request())]

            task = asyncio.create_task(consume())
            async with asyncio.timeout(5):
                while not list(Path(directory).glob("call-*.pid")):
                    await asyncio.sleep(0.01)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertFalse(list(Path(directory).glob("call-*.pid")))
