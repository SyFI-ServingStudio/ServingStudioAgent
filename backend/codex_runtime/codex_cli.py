"""Subprocess lifecycle for one `codex exec` role call."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..logging_config import compact_text, log_event
from .codex_command import build_codex_exec_command
from .config import (
    CODEX_DOCKER_GID,
    CODEX_DOCKER_HOME,
    CODEX_DOCKER_UID,
    CODEX_IDLE_TIMEOUT,
    DEFAULT_CODEX_SERVICE_TIER,
    LOG,
)
from .exec_types import CodexEvent, CodexExecRequest
from .output_collector import CodexOutputCollector

# How often the pre-first-output line refreshes its elapsed counter.
FIRST_OUTPUT_TICK_SECONDS = 5.0


async def run_codex(
    container: str,
    prompt: str,
    *,
    label: str,
    workspace_id: str,
    conversation_id: str,
    turn_id: str,
    model_id: str,
    effort: str,
    service_tier: str = DEFAULT_CODEX_SERVICE_TIER,
    session_id: str | None = None,
    output_schema: str | None = None,
) -> AsyncIterator[CodexEvent]:
    """Run one `codex exec` or `codex exec resume` call and stream UI events."""
    request = CodexExecRequest(
        container=container,
        prompt=prompt,
        label=label,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        model_id=model_id,
        effort=effort,
        service_tier=service_tier,
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
        output = CodexOutputCollector(self.request)

        # Opens this role's interrupt blind window — see `_stream_process_output`
        # for the pairing `role_ready`. Published before anything here can
        # suspend, because it is what tells a canceller that this role's rollout
        # does not hold the handoff prompt yet. Emitted after startup instead,
        # process creation and prompt delivery — both of which suspend, and the
        # second of which is the handoff itself — sat inside the *previous*
        # role's ready window, where a Stop is authorised at once.
        yield {"kind": "role_start", "role": self.request.label}

        # Codex says nothing until its first item, so without this the UI would
        # still read "checking Docker Codex container..." through CLI startup,
        # session load and the model's time to first token — usually the longest
        # stretch of the wait, and the one users most want narrated.
        yield self._waiting_event(0.0, session_ready=False)

        wait = _FirstOutputWait(started)
        process: asyncio.subprocess.Process | None = None
        stderr_task: asyncio.Task[None] | None = None
        try:
            process = await self._start_process()
            await self._write_prompt(process)
            stderr_task = asyncio.create_task(self._drain_stderr(process, output))
            output.prime_rollout_offset()
            async for event in self._stream_process_output(process, output, wait):
                yield event
        except asyncio.CancelledError:
            # `process` is None when the cancel arrived before it existed, and
            # otherwise it may be only half set up — spawned, with its prompt
            # partly written. It still has to be stopped: a container process
            # nobody is reading from outlives the turn that started it.
            if process is not None:
                await self._handle_cancelled(process)
            raise
        finally:
            if stderr_task is not None:
                await self._finish_stderr_task(stderr_task)

        # A call can produce nothing a reader sees until its final answer — a
        # one-line reply with no tool use is the common case. The rollout holds
        # this call's prompt by then, so the blind window has to close here or
        # an armed interrupt would wait for an event that never comes.
        if wait.note_call_end():
            yield {"kind": "role_ready", "role": self.request.label}

        if output.timed_out:
            yield output.timeout_event()

        # Emit usage before final: the caller attaches it to the phase that is
        # closing, then the final/implementer event caps the phase.
        duration_ms = int((asyncio.get_event_loop().time() - started) * 1000)
        yield output.usage_event(duration_ms)
        yield output.final_event(process.returncode)

    def _waiting_event(self, elapsed: float, session_ready: bool) -> CodexEvent:
        """The "nothing has arrived yet" line, ticking so the wait reads as alive."""
        model = self.request.model_id.rsplit("/", 1)[-1]
        stage = (
            "waiting for first output from" if session_ready else "starting Codex with"
        )
        age = f" ({elapsed:.0f}s)" if elapsed >= 1 else ""
        return {
            "kind": "tool_call",
            "text": (
                f"{self.request.label}: {stage} {model} · {self.request.effort}"
                f"{' · fast' if self.request.service_tier == 'fast' else ''}{age}..."
            ),
        }

    def _log_start(self) -> None:
        log_event(
            LOG,
            "codex.start",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.label,
            model=self.request.model_id,
            effort=self.request.effort,
            service_tier=self.request.service_tier,
            container=self.request.container,
            resume=self.request.is_resume,
            codex_session_id=self.request.session_id,
            output_schema=self.request.output_schema
            if self.request.schema_arg_used
            else "",
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
        wait: _FirstOutputWait,
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
                    for noted in self._noted(wait, event):
                        yield noted
                if not done:
                    if wait.due():
                        yield self._waiting_event(wait.tick(), wait.session_ready)
                    continue

                raw = read_task.result()
                if not raw:
                    break
                idle_deadline = _new_idle_deadline()
                stdout_buffer += raw
                complete_lines, stdout_buffer = _split_stdout_lines(stdout_buffer)
                for raw_line in complete_lines:
                    for event in output.events_from_stdout_line(raw_line):
                        for noted in self._noted(wait, event):
                            yield noted
                read_task = asyncio.create_task(process.stdout.read(8192))
        finally:
            await self._cancel_read_task(read_task)

        # These tail drains can carry a call's only output — a short answer that
        # arrived in one read, say — so they go through `_noted` too, or the
        # blind window would never close for that call.
        if stdout_buffer.strip():
            for event in output.events_from_stdout_line(stdout_buffer):
                for noted in self._noted(wait, event):
                    yield noted
        for event in output.poll_rollout_intermediate_outputs():
            for noted in self._noted(wait, event):
                yield noted
        if output.timed_out:
            process.kill()
        await process.wait()
        # Codex can persist task_complete while shutting down after stdout has
        # already reached EOF. Refresh once more so final_event can recover the
        # durable handoff instead of reporting a false empty result.
        for event in output.poll_rollout_intermediate_outputs():
            for noted in self._noted(wait, event):
                yield noted

    def _noted(self, wait: _FirstOutputWait, event: CodexEvent) -> Iterator[CodexEvent]:
        """Yield `event`, preceded by `role_ready` if it is this call's first output.

        `role_ready` closes the interrupt blind window: until a role has produced
        something, the prompt that carries the handoff (a delegated task, or an
        implementer summary on its way back) is not yet durable in its rollout,
        so interrupting there would drop it. Emitted before the event itself so a
        client that arms an interrupt sees the window close first, and exactly
        once — `_FirstOutputWait.note` latches.
        """
        if wait.note(event):
            yield {"kind": "role_ready", "role": self.request.label}
        yield event

    async def _cancel_read_task(self, read_task: asyncio.Task[bytes]) -> None:
        if read_task.done():
            return
        read_task.cancel()
        await _await_stopped(read_task)

    async def _handle_cancelled(self, process: asyncio.subprocess.Process) -> None:
        log_event(
            LOG,
            "codex.cancelled",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.label,
            pid=process.pid,
        )
        # Killing the host-side `docker exec` client does not stop the process it
        # launched in the container. Interrupt the one Codex call allowed by the
        # per-conversation lock first, matching terminal Ctrl+C semantics.
        await self._signal_container_codex("INT")
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                await self._signal_container_codex("TERM")
                process.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), timeout=5)

    async def _signal_container_codex(self, signal_name: str) -> None:
        signal_process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            self.request.container,
            "sh",
            "-c",
            f"pkill -{signal_name} -f '[c]odex exec' || true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        with contextlib.suppress(Exception):
            await asyncio.wait_for(signal_process.wait(), timeout=5)

    async def _finish_stderr_task(self, stderr_task: asyncio.Task[None]) -> None:
        if not stderr_task.done():
            stderr_task.cancel()
        await _await_stopped(stderr_task)


async def _await_stopped(task: asyncio.Task[Any]) -> None:
    """Wait for a helper task we have cancelled, without losing a Stop.

    Awaiting a cancelled task raises `CancelledError` — and so does being
    cancelled *while* awaiting it. Same exception, opposite meanings, and
    suppressing both is what a bare `suppress(BaseException)` here used to do.

    The cost is not theoretical. Both call sites sit in a `finally`, which is
    exactly where a Stop arriving at the end of a role lands. Swallowed, the
    turn walks on into its next role while the conversation has already recorded
    the cancellation as delivered — so every later Stop is a no-op and the turn
    can no longer be stopped at all.

    `cancelling()` counts cancellations aimed at *this* task, which is the one
    thing that tells the two apart. A failure of the helper itself is still
    dropped: nothing reads its result, and it says nothing about the turn.
    """
    try:
        await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling() > 0:
            raise
    except Exception:
        pass


class _FirstOutputWait:
    """Ticks the waiting line until Codex produces something a user can read.

    The session event is Codex answering the CLI, not the model, so it only
    changes the wording: everything after it is time to first token.
    """

    def __init__(self, started: float) -> None:
        self.started = started
        self.session_ready = False
        self.first_output_seen = False
        self.next_tick = started + FIRST_OUTPUT_TICK_SECONDS

    def note(self, event: CodexEvent) -> bool:
        """Record one event; return True only on the call that sees first output.

        The `session` event is Codex answering the CLI rather than the model, so
        it does not count as output — the rollout does not yet hold this call's
        prompt. Callers use the True edge to emit `role_ready`.
        """
        if event.get("kind") == "session":
            self.session_ready = True
            return False
        if self.first_output_seen:
            return False
        self.first_output_seen = True
        return True

    def note_call_end(self) -> bool:
        """Latch at the end of a silent call; True if that is what closes it."""
        if self.first_output_seen:
            return False
        self.first_output_seen = True
        return True

    def due(self) -> bool:
        if self.first_output_seen:
            return False
        return asyncio.get_event_loop().time() >= self.next_tick

    def tick(self) -> float:
        now = asyncio.get_event_loop().time()
        self.next_tick = now + FIRST_OUTPUT_TICK_SECONDS
        return now - self.started


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
