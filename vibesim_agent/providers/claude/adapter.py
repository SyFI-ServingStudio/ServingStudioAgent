"""One Claude stream-json invocation with independent configuration and cleanup."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping

from ...runtime.invocation import InvocationHome
from ...runtime.process import ProcessStream
from ..base import AgentRequest
from .command import ClaudeCommand
from .events import ClaudeOutputCollector


class ClaudeAdapter:
    adapter_id = "claude"

    def __init__(
        self,
        command: ClaudeCommand,
        *,
        home: Callable[[AgentRequest], InvocationHome],
        idle_timeout: float,
        logger: logging.Logger,
        poll_interval: float = 5,
        process_environment: Mapping[str, str] | None = None,
    ):
        self.command = command
        self.home = home
        self.idle_timeout = idle_timeout
        self.logger = logger
        self.poll_interval = poll_interval
        self.process_environment = (
            dict(process_environment) if process_environment is not None else None
        )

    async def run(self, request: AgentRequest) -> AsyncIterator[dict]:
        yield {"kind": "role_start", "role": request.role.value}
        home = self.home(request)
        invocation = request.execution.invocation(
            execution_id=request.execution_id,
            home=home,
            logger=self.logger,
            label="Claude",
            process_environment=self.process_environment,
        )
        collector = ClaudeOutputCollector(
            request, schema=self.command.output_schema(request)
        )
        call = ProcessStream(
            self.command.build_tracked(
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
        )
        started = asyncio.get_running_loop().time()
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
                            for event in collector.events_from_stdout_line(bytes(line)):
                                yield event
                        if len(buffer) > 16 * 1024 * 1024:
                            raise RuntimeError("Claude stream event exceeds 16 MiB")
                    elif output.stream == "tick" and not collector.ready:
                        elapsed = asyncio.get_running_loop().time() - started
                        yield {
                            "kind": "tool_call",
                            "text": f"{request.role.value}: waiting for Claude ({elapsed:.0f}s)...",
                        }
                if buffer.strip():
                    for event in collector.events_from_stdout_line(bytes(buffer)):
                        yield event
                if call.timed_out:
                    invocation.mark_cancelled()
            completed = True
            assert call.process is not None
            for event in collector.finish(
                call.process.returncode,
                int((asyncio.get_running_loop().time() - started) * 1000),
                timed_out=call.timed_out,
            ):
                yield event
        except BaseException:
            if not completed:
                invocation.mark_cancelled()
            raise
        finally:
            invocation.remove_pid()
