"""Subprocess lifecycle for one `codex exec` role call."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from .codex_command import build_codex_exec_command
from .config import (
    CODEX_IDLE_TIMEOUT,
    CODEX_DOCKER_GID,
    CODEX_DOCKER_HOME,
    CODEX_DOCKER_UID,
    LOG,
)
from .exec_types import CodexEvent, CodexExecRequest
from .output_collector import CodexOutputCollector
from ..logging_config import compact_text, log_event


async def run_codex(
    container: str,
    prompt: str,
    *,
    label: str,
    conversation_id: str,
    turn_id: str,
    session_id: str | None = None,
    output_schema: str | None = None,
) -> AsyncIterator[CodexEvent]:
    """Run one `codex exec` or `codex exec resume` call and stream UI events."""
    request = CodexExecRequest(
        container=container,
        prompt=prompt,
        label=label,
        conversation_id=conversation_id,
        turn_id=turn_id,
        session_id=session_id,
        output_schema=output_schema,
    )
    async for event in CodexExecCall(request).run():
        yield event


class CodexExecCall:
    def __init__(self, request: CodexExecRequest) -> None:
        self.request = request

    async def run(self) -> AsyncIterator[CodexEvent]:
        self._log_start()
        started = asyncio.get_event_loop().time()
        process = await self._start_process()
        output = CodexOutputCollector(self.request)
        await self._write_prompt(process)
        stderr_task = asyncio.create_task(self._drain_stderr(process, output))

        try:
            output.prime_rollout_offset()
            async for event in self._stream_process_output(process, output):
                yield event
        except asyncio.CancelledError:
            await self._handle_cancelled(process)
            raise
        finally:
            await self._finish_stderr_task(stderr_task)

        if output.timed_out:
            yield output.timeout_event()

        # Emit usage before final: the caller attaches it to the phase that is
        # closing, then the final/implementer event caps the phase.
        duration_ms = int((asyncio.get_event_loop().time() - started) * 1000)
        yield output.usage_event(duration_ms)
        yield output.final_event(process.returncode)

    def _log_start(self) -> None:
        log_event(
            LOG,
            "codex.start",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.label,
            container=self.request.container,
            resume=self.request.is_resume,
            codex_session_id=self.request.session_id,
            output_schema=self.request.output_schema if self.request.schema_arg_used else "",
            output_schema_requested=bool(self.request.output_schema),
            output_schema_arg_used=self.request.schema_arg_used,
            uid=CODEX_DOCKER_UID,
            gid=CODEX_DOCKER_GID,
            docker_home=CODEX_DOCKER_HOME,
            prompt_len=len(self.request.prompt),
            prompt_preview=compact_text(self.request.prompt),
        )

    async def _start_process(self) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            *build_codex_exec_command(self.request),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _write_prompt(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdin is not None
        process.stdin.write(self.request.prompt.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

    async def _drain_stderr(
        self,
        process: asyncio.subprocess.Process,
        output: CodexOutputCollector,
    ) -> None:
        assert process.stderr is not None
        async for line in process.stderr:
            output.append_stderr_line(line)

    async def _stream_process_output(
        self,
        process: asyncio.subprocess.Process,
        output: CodexOutputCollector,
    ) -> AsyncIterator[CodexEvent]:
        assert process.stdout is not None
        stdout_buffer = b""
        read_task = asyncio.create_task(process.stdout.read(8192))
        idle_deadline = _new_idle_deadline()

        try:
            while True:
                if _timed_out(idle_deadline):
                    output.mark_timed_out()
                    break

                done, _pending = await asyncio.wait(
                    {read_task},
                    timeout=_poll_interval(idle_deadline),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                rollout_events = output.poll_rollout_intermediate_outputs()
                if rollout_events:
                    idle_deadline = _new_idle_deadline()
                for event in rollout_events:
                    yield event
                if not done:
                    continue

                raw = read_task.result()
                if not raw:
                    break
                idle_deadline = _new_idle_deadline()
                stdout_buffer += raw
                complete_lines, stdout_buffer = _split_stdout_lines(stdout_buffer)
                for raw_line in complete_lines:
                    for event in output.events_from_stdout_line(raw_line):
                        yield event
                read_task = asyncio.create_task(process.stdout.read(8192))
        finally:
            await self._cancel_read_task(read_task)

        if stdout_buffer.strip():
            for event in output.events_from_stdout_line(stdout_buffer):
                yield event
        for event in output.poll_rollout_intermediate_outputs():
            yield event
        if output.timed_out:
            process.kill()
        await process.wait()

    async def _cancel_read_task(self, read_task: asyncio.Task[bytes]) -> None:
        if read_task.done():
            return
        read_task.cancel()
        with contextlib.suppress(BaseException):
            await read_task

    async def _handle_cancelled(self, process: asyncio.subprocess.Process) -> None:
        log_event(
            LOG,
            "codex.cancelled",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.label,
            pid=process.pid,
        )
        if process.returncode is None:
            process.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=5)

    async def _finish_stderr_task(self, stderr_task: asyncio.Task[None]) -> None:
        if not stderr_task.done():
            stderr_task.cancel()
        with contextlib.suppress(BaseException):
            await stderr_task


def _new_idle_deadline() -> float:
    return asyncio.get_event_loop().time() + CODEX_IDLE_TIMEOUT


def _timed_out(deadline: float) -> bool:
    return asyncio.get_event_loop().time() >= deadline


def _poll_interval(deadline: float) -> float:
    remaining = deadline - asyncio.get_event_loop().time()
    return min(1.0, max(0.0, remaining))


def _split_stdout_lines(stdout_buffer: bytes) -> tuple[list[bytes], bytes]:
    if b"\n" not in stdout_buffer:
        return [], stdout_buffer
    *complete_lines, trailing_fragment = stdout_buffer.split(b"\n")
    return complete_lines, trailing_fragment
