"""Lifecycle of a Claude role call, including container-side interruption."""

import asyncio
import contextlib
from collections.abc import AsyncIterator

from ..logging_config import log_event
from .claude_command import build_claude_exec_command, pid_file
from .claude_events import ClaudeOutputCollector
from .config import CODEX_IDLE_TIMEOUT, LOG, role_codex_home_for
from .exec_types import CodexEvent, CodexExecRequest


async def signal_claude(container: str, signal: str, path: str) -> None:
    if signal not in {"INT", "TERM", "KILL"}:
        raise ValueError("unsupported process signal")
    process = await asyncio.create_subprocess_exec(
        "docker",
        "exec",
        container,
        "sh",
        "-c",
        'if [ -f "$1" ]; then kill -"$2" "$(cat "$1")" 2>/dev/null || true; fi',
        "vibesim-signal",
        path,
        signal,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()


async def stop_claude(
    process: asyncio.subprocess.Process, request: CodexExecRequest
) -> None:
    # Killing docker exec alone leaves the actual agent running in Docker.
    for signal in ("INT", "TERM", "KILL"):
        await signal_claude(request.container, signal, pid_file(request))
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
            return
        except TimeoutError:
            pass
    if process.returncode is None:
        process.kill()
        await process.wait()


async def run_claude(
    container: str, prompt: str, **options
) -> AsyncIterator[CodexEvent]:
    request = CodexExecRequest(container=container, prompt=prompt, **options)
    collector = ClaudeOutputCollector(request)
    loop = asyncio.get_running_loop()
    started = loop.time()
    process = await asyncio.create_subprocess_exec(
        *build_claude_exec_command(request),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr_tail = bytearray()

    async def drain_stderr() -> None:
        assert process.stderr is not None
        while chunk := await process.stderr.read(8192):
            stderr_tail.extend(chunk)
            del stderr_tail[:-8192]

    stderr_task = asyncio.create_task(drain_stderr())
    read_task = None
    wait_task = None
    timed_out = False
    try:
        yield {"kind": "role_start", "role": request.label}
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Read the CLI's terminal failure instead of losing it.
        finally:
            process.stdin.close()
        buffer = b""
        deadline = loop.time() + CODEX_IDLE_TIMEOUT
        read_task = asyncio.create_task(process.stdout.read(8192))
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                timed_out = True
                break
            done, _ = await asyncio.wait({read_task}, timeout=min(5, remaining))
            if not done:
                if not collector.ready:
                    yield {
                        "kind": "tool_call",
                        "text": f"{request.label}: waiting for Claude ({loop.time() - started:.0f}s)...",
                    }
                continue
            chunk = read_task.result()
            if not chunk:
                if buffer.strip():
                    for event in collector.events_from_stdout_line(buffer):
                        yield event
                break
            deadline = loop.time() + CODEX_IDLE_TIMEOUT
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                for event in collector.events_from_stdout_line(line):
                    yield event
            # Large tool payloads are legitimate, but a broken CLI must not
            # grow backend memory without bound on an unterminated line.
            if len(buffer) > 16 * 1024 * 1024:
                raise RuntimeError("Claude stream event exceeds 16 MiB")
            read_task = asyncio.create_task(process.stdout.read(8192))
        if not timed_out:
            wait_task = asyncio.create_task(process.wait())
            done, _ = await asyncio.wait(
                {wait_task}, timeout=max(0, deadline - loop.time())
            )
            timed_out = not done
        if timed_out:
            await stop_claude(process, request)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stderr_task, timeout=5)
        log_event(
            LOG,
            "claude.finish",
            conversation_id=request.conversation_id,
            role=request.label,
            model=request.model_id,
            returncode=process.returncode,
            timed_out=timed_out,
            has_result=collector.result is not None,
            stderr_bytes=len(stderr_tail),
        )
        for event in collector.finish(
            process.returncode or 0,
            int((loop.time() - started) * 1000),
            timed_out=timed_out,
        ):
            yield event
    finally:
        if process.returncode is None:
            await stop_claude(process, request)
        for task in (read_task, wait_task, stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        # Each invocation has a unique file, so cancellation during startup can
        # never signal a previous role call. Keep the durable home bounded.
        local_pid_file = (
            role_codex_home_for(
                request.workspace_id, request.conversation_id, request.label
            )
            / "claude"
            / f"call-{request.execution_id}.pid"
        )
        with contextlib.suppress(OSError):
            local_pid_file.unlink(missing_ok=True)
