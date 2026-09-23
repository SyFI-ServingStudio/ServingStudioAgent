"""One Codex role invocation using shared pipes and private durable output."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping

from ...runtime.invocation import InvocationHome as CodexHome
from ...runtime.process import ProcessStream
from ..base import AgentRequest
from .collector import CodexOutputCollector
from .command import CodexCommand


class CodexAdapter:
    adapter_id = "codex"

    def __init__(
        self,
        command: CodexCommand,
        *,
        home: Callable[[AgentRequest], CodexHome],
        idle_timeout: float,
        logger: logging.Logger,
        poll_interval: float = 5,
        command_for_home: Callable[[CodexHome], CodexCommand] | None = None,
        process_environment: Mapping[str, str] | None = None,
    ):
        self.command = command
        self.home = home
        self.idle_timeout = idle_timeout
        self.logger = logger
        self.poll_interval = poll_interval
        self.command_for_home = command_for_home
        self.process_environment = (
            dict(process_environment) if process_environment is not None else None
        )

    async def run(self, request: AgentRequest) -> AsyncIterator[dict]:
        yield {"kind": "role_start", "role": request.role.value}
        home = self.home(request)
        command = (
            self.command_for_home(home)
            if self.command_for_home is not None
            else self.command
        )
        invocation = request.execution.invocation(
            execution_id=request.execution_id,
            home=home,
            logger=self.logger,
            label="Codex",
            process_environment=self.process_environment,
        )
        collector = CodexOutputCollector(
            request, home=home.host, idle_timeout=self.idle_timeout, logger=self.logger
        )
        collector.prime_rollout_offset()
        started = asyncio.get_running_loop().time()
        call = ProcessStream(
            command.build_tracked(
                request,
                home=request.execution.home_path(home),
                pid_file=invocation.pid_file,
            ),
            input_data=request.prompt.encode("utf-8"),
            idle_timeout=self.idle_timeout,
            stop=invocation.stop,
            poll_interval=self.poll_interval,
            stop_timeout=request.execution.stop_timeout,
            environment=request.execution.spawn_environment(self.process_environment),
            cwd=request.execution.cwd,
            start_new_session=request.execution.start_new_session,
            kill=request.execution.kill,
            started=invocation.record,
        )
        ready = False

        def noted(events):
            nonlocal ready
            if not ready and collector.ready:
                ready = True
                yield {"kind": "role_ready", "role": request.role.value}
            yield from events

        buffer = bytearray()
        completed = False
        try:
            async with call:
                async for output in call.events():
                    if output.stream == "stdout":
                        buffer.extend(output.data)
                        while b"\n" in buffer:
                            line, _, tail = buffer.partition(b"\n")
                            buffer = bytearray(tail)
                            for event in noted(
                                collector.events_from_stdout_line(bytes(line))
                            ):
                                yield event
                        if len(buffer) > 16 * 1024 * 1024:
                            raise RuntimeError("Codex stream event exceeds 16 MiB")
                    elif output.stream == "stderr":
                        collector.append_stderr_line(output.data)
                    events = collector.poll_rollout_intermediate_outputs()
                    if events:
                        call.touch()
                    for event in noted(events):
                        yield event
                    if output.stream == "tick" and not ready:
                        elapsed = asyncio.get_running_loop().time() - started
                        yield {
                            "kind": "tool_call",
                            "text": f"{request.role.value}: waiting for Codex ({elapsed:.0f}s)...",
                        }
                if buffer.strip():
                    for event in noted(
                        collector.events_from_stdout_line(bytes(buffer))
                    ):
                        yield event
                if call.timed_out:
                    invocation.mark_cancelled()
            # Codex may persist task_complete while shutting down after stdout EOF.
            completed = True
            for event in noted(collector.poll_rollout_intermediate_outputs()):
                yield event
            if call.timed_out:
                collector.mark_timed_out()
                yield collector.timeout_event()
            final = collector.final_event(call.process.returncode)
            for event in noted([]):
                yield event
            yield collector.usage_event(
                int((asyncio.get_running_loop().time() - started) * 1000)
            )
            yield final
        except BaseException:
            if not completed:
                invocation.mark_cancelled()
            raise
        finally:
            invocation.remove_pid()
