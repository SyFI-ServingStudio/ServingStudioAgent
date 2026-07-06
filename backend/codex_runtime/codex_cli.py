"""Low-level `codex exec` subprocess runner."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

from .codex_events import (
    _codex_stderr_for_error,
    _find_rollout_file,
    _scan_rollout_agent_messages,
    _translate,
)
from .config import (
    CODEX_CALL_TIMEOUT,
    CODEX_DOCKER_DG_USE_LOCAL_VERSION,
    CODEX_DOCKER_GID,
    CODEX_DOCKER_GPUS,
    CODEX_DOCKER_HOME,
    CODEX_DOCKER_UID,
    CODEX_DOCKER_USER,
    CODEX_DOCKER_UV_CACHE_DIR,
    CODEX_DOCKER_UV_PROJECT_ENVIRONMENT,
    CODEX_MODEL,
    LOG,
    MAIN_LOCK_SHA,
)
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
) -> AsyncIterator[dict[str, str]]:
    cmd = [
        "docker",
        "exec",
        "-i",
        "-u",
        f"{CODEX_DOCKER_UID}:{CODEX_DOCKER_GID}",
        "-e",
        f"HOME={CODEX_DOCKER_HOME}",
        "-e",
        f"USER={CODEX_DOCKER_USER}",
        "-e",
        f"LOGNAME={CODEX_DOCKER_USER}",
        "-e",
        f"UV_PROJECT_ENVIRONMENT={CODEX_DOCKER_UV_PROJECT_ENVIRONMENT}",
        "-e",
        f"UV_CACHE_DIR={CODEX_DOCKER_UV_CACHE_DIR}",
        "-e",
        f"MLSIM_EXPECTED_LOCK_SHA={MAIN_LOCK_SHA}",
        "-e",
        f"DG_USE_LOCAL_VERSION={CODEX_DOCKER_DG_USE_LOCAL_VERSION}",
        "-e",
        f"CODEX_DOCKER_GPUS={CODEX_DOCKER_GPUS}",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-w",
        "/workspace",
        container,
        "codex",
        "exec",
    ]
    codex_options = [
        "-m",
        CODEX_MODEL,
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
    ]
    if session_id:
        # `codex exec resume` only reads stdin when the prompt argument is "-".
        cmd.append("resume")
        cmd.extend(codex_options)
        cmd.extend([session_id, "-"])
    else:
        if output_schema:
            codex_options.extend(["--output-schema", output_schema])
        cmd.extend(codex_options)
    schema_arg_used = bool(output_schema and not session_id)
    log_event(
        LOG,
        "codex.start",
        conversation_id=conversation_id,
        turn_id=turn_id,
        role=label,
        container=container,
        resume=bool(session_id),
        codex_session_id=session_id,
        output_schema=output_schema if schema_arg_used else "",
        output_schema_requested=bool(output_schema),
        output_schema_arg_used=schema_arg_used,
        uid=CODEX_DOCKER_UID,
        gid=CODEX_DOCKER_GID,
        docker_home=CODEX_DOCKER_HOME,
        prompt_len=len(prompt),
        prompt_preview=compact_text(prompt),
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert proc.stdin is not None
    proc.stdin.write(prompt.encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()

    stderr_chunks: list[str] = []

    async def _drain_stderr() -> None:
        assert proc.stderr is not None
        async for line in proc.stderr:
            stderr_chunks.append(line.decode("utf-8", "replace"))

    err_task = asyncio.create_task(_drain_stderr())
    final_text: str | None = None
    seen_intermediate_outputs: set[tuple[str, str]] = set()
    current_session_id = session_id
    rollout_file: Path | None = None
    rollout_offset = 0
    if current_session_id:
        rollout_file = _find_rollout_file(conversation_id, current_session_id)
        if rollout_file is not None:
            with contextlib.suppress(OSError):
                rollout_offset = rollout_file.stat().st_size
    loop = asyncio.get_event_loop()
    deadline = loop.time() + CODEX_CALL_TIMEOUT
    timed_out = False

    def _poll_rollout_intermediate_outputs() -> list[dict[str, str]]:
        nonlocal rollout_file, rollout_offset
        if not current_session_id:
            return []
        if rollout_file is None:
            rollout_file = _find_rollout_file(conversation_id, current_session_id)
            if rollout_file is None:
                return []
        messages, rollout_offset = _scan_rollout_agent_messages(rollout_file, rollout_offset)
        notes = []
        for text, phase in messages:
            if phase != "commentary":
                continue
            note_text = text.strip()
            if not note_text:
                continue
            note_key = (label, note_text)
            if note_key in seen_intermediate_outputs:
                continue
            seen_intermediate_outputs.add(note_key)
            log_event(
                LOG,
                "codex.intermediate_output",
                conversation_id=conversation_id,
                turn_id=turn_id,
                role=label,
                text=note_text,
                source="rollout",
            )
            notes.append({"kind": "intermediate_output", "role": label, "text": note_text})
        return notes

    def _translate_stdout_line(raw_line: bytes) -> list[dict[str, str]]:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line:
            return []
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return []
        return _translate(ev)

    async def _emit_translated(out: dict[str, str]) -> AsyncIterator[dict[str, str]]:
        nonlocal current_session_id, final_text
        if out["kind"] == "session":
            current_session_id = out["session_id"]
            log_event(
                LOG,
                "codex.session",
                conversation_id=conversation_id,
                turn_id=turn_id,
                role=label,
                codex_session_id=out["session_id"],
            )
            yield {
                "kind": "session",
                "role": label,
                "session_id": out["session_id"],
            }
        elif out["kind"] == "agent_text":
            text = out.get("text") or ""
            phase = out.get("phase") or ""
            if phase == "commentary":
                note_text = text.strip()
                if not note_text:
                    return
                note_key = (label, note_text)
                if note_key in seen_intermediate_outputs:
                    return
                seen_intermediate_outputs.add(note_key)
                log_event(
                    LOG,
                    "codex.intermediate_output",
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    role=label,
                    text=note_text,
                    source="stdout",
                )
                yield {"kind": "intermediate_output", "role": label, "text": note_text}
            elif text.strip():
                final_text = text.strip()
        else:
            progress_text = out.get("text", "")
            log_event(
                LOG,
                "codex.progress",
                conversation_id=conversation_id,
                turn_id=turn_id,
                role=label,
                text=progress_text,
            )
            yield {"kind": "progress", "text": f"{label}: {progress_text}"}

    try:
        assert proc.stdout is not None
        stdout_buffer = b""
        read_task = asyncio.create_task(proc.stdout.read(8192))
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                timed_out = True
                break
            done, _pending = await asyncio.wait(
                {read_task},
                timeout=min(1.0, remaining),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for note in _poll_rollout_intermediate_outputs():
                yield note
            if not done:
                continue
            raw = read_task.result()
            if not raw:
                break
            stdout_buffer += raw
            while b"\n" in stdout_buffer:
                raw_line, stdout_buffer = stdout_buffer.split(b"\n", 1)
                for out in _translate_stdout_line(raw_line):
                    async for emitted in _emit_translated(out):
                        yield emitted
            read_task = asyncio.create_task(proc.stdout.read(8192))

        if not read_task.done():
            read_task.cancel()
            with contextlib.suppress(BaseException):
                await read_task
        if stdout_buffer.strip():
            for out in _translate_stdout_line(stdout_buffer):
                async for emitted in _emit_translated(out):
                    yield emitted
        for note in _poll_rollout_intermediate_outputs():
            yield note
        if timed_out:
            proc.kill()
        await proc.wait()
    except asyncio.CancelledError:
        log_event(
            LOG,
            "codex.cancelled",
            conversation_id=conversation_id,
            turn_id=turn_id,
            role=label,
            pid=proc.pid,
        )
        if proc.returncode is None:
            proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=5)
        raise
    finally:
        if not err_task.done():
            err_task.cancel()
        with contextlib.suppress(BaseException):
            await err_task

    if timed_out:
        log_event(
            LOG,
            "codex.timeout",
            conversation_id=conversation_id,
            turn_id=turn_id,
            role=label,
            timeout_s=CODEX_CALL_TIMEOUT,
        )
        yield {"kind": "error", "text": f"{label}: codex call timed out after {CODEX_CALL_TIMEOUT:.0f}s"}

    raw_stderr_text = "".join(stderr_chunks).strip()
    stderr_text = _codex_stderr_for_error(
        raw_stderr_text,
        returncode=proc.returncode,
        has_final_text=final_text is not None,
    )
    if final_text is None:
        if proc.returncode not in (0, None) and stderr_text:
            final_text = f"({label} exited {proc.returncode})\n\n```\n{stderr_text[-1500:]}\n```"
        elif stderr_text:
            final_text = f"({label} produced no final text)\n\n```\n{stderr_text[-1500:]}\n```"
        else:
            final_text = f"({label} produced no final text)"

    log_event(
        LOG,
        "codex.final",
        conversation_id=conversation_id,
        turn_id=turn_id,
        role=label,
        returncode=proc.returncode,
        final_len=len(final_text),
        final_preview=compact_text(final_text),
        stderr_len=len(stderr_text),
        stderr_tail=stderr_text[-500:] if stderr_text else "",
        raw_stderr_len=len(raw_stderr_text),
    )
    yield {"kind": "final", "text": final_text}
