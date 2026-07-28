"""Codex JSON event and rollout-log translation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import codex_home_for


def _describe_item(item: dict[str, Any]) -> str:
    itype = item.get("type", "item")
    if itype in ("command_execution", "command", "local_shell_call"):
        cmd = item.get("command") or item.get("cmd") or item.get("action") or ""
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        return f"$ {str(cmd)[:240]}"
    if itype == "reasoning":
        return "...thinking"
    if itype in ("file_change", "patch", "apply_patch"):
        return "editing files..."
    if itype == "mcp_tool_call":
        return f"tool: {item.get('tool') or item.get('name') or ''}"
    text = item.get("text") or item.get("summary") or itype
    return str(text)[:240]


def _assistant_message_from_payload(payload: dict[str, Any]) -> tuple[str, str] | None:
    """Extract assistant text plus Codex phase from known JSON event payloads."""
    if payload.get("type") == "agent_message":
        text = payload.get("message") or payload.get("text") or ""
        return str(text), str(payload.get("phase") or "")

    if payload.get("type") != "message" or payload.get("role") != "assistant":
        return None

    content = payload.get("content") or []
    parts: list[str] = []
    if isinstance(content, list):
        for entry in content:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if text is None:
                text = entry.get("output_text") or entry.get("input_text")
            if text is not None:
                parts.append(str(text))
    elif isinstance(content, str):
        parts.append(content)

    return "".join(parts), str(payload.get("phase") or "")


def _find_rollout_file(
    workspace_id: str, conversation_id: str, session_id: str
) -> Path | None:
    sessions_dir = codex_home_for(workspace_id, conversation_id) / "sessions"
    if not sessions_dir.exists():
        return None
    candidates = []
    for path in sessions_dir.rglob(f"*{session_id}*.jsonl"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _scan_rollout_agent_messages(
    rollout_file: Path,
    offset: int,
) -> tuple[list[tuple[str, str]], int]:
    try:
        size = rollout_file.stat().st_size
    except OSError:
        return [], offset
    if offset > size:
        offset = 0

    messages: list[tuple[str, str]] = []
    try:
        with rollout_file.open("rb") as file:
            file.seek(offset)
            for raw_line in file:
                try:
                    event = json.loads(raw_line.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                if event.get("type") not in ("event_msg", "response_item"):
                    continue
                payload = event.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                assistant_message = _assistant_message_from_payload(payload)
                if assistant_message is not None:
                    messages.append(assistant_message)
            return messages, file.tell()
    except OSError:
        return [], offset


def unwrap_commentary(text: str) -> str:
    """Clean an orchestrator commentary note for display.

    The orchestrator sometimes narrates by emitting its whole decision envelope
    (``{"action": ..., "message": ..., "task": ...}``) into the commentary
    channel instead of plain prose. Show the human ``message`` when present; drop
    a bare decision envelope (its task/decision is surfaced as its own card);
    otherwise leave the text untouched.
    """
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return text
    try:
        payload = json.loads(stripped)
    except ValueError:
        return text
    if not isinstance(payload, dict):
        return text
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    if "action" in payload or "task" in payload:
        return ""
    return text


def _scan_rollout_last_token_usage(rollout_file: Path) -> dict[str, int] | None:
    """Return the most recent cumulative ``total_token_usage`` in a rollout log.

    Codex writes ``token_count`` events (``event_msg`` payloads) whose
    ``info.total_token_usage`` is the *session-cumulative* usage. Early events can
    carry ``info: null`` (rate-limit-only pings) and are skipped. The last usable
    one before a call is the baseline; the last after is the end — so a per-call
    delta is ``end - baseline``.
    """
    try:
        with rollout_file.open("rb") as file:
            latest: dict[str, int] | None = None
            for raw_line in file:
                try:
                    event = json.loads(raw_line.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "event_msg":
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict) or payload.get("type") != "token_count":
                    continue
                info = payload.get("info")
                if not isinstance(info, dict):
                    continue
                usage = info.get("total_token_usage")
                if isinstance(usage, dict):
                    latest = usage
            return latest
    except OSError:
        return None


def _translate(ev: dict[str, Any]) -> list[dict[str, str]]:
    etype = ev.get("type")
    if etype == "thread.started" and ev.get("thread_id"):
        return [{"kind": "session", "session_id": str(ev["thread_id"])}]
    if etype in ("event_msg", "response_item"):
        payload = ev.get("payload") or {}
        if isinstance(payload, dict):
            assistant_message = _assistant_message_from_payload(payload)
            if assistant_message is not None:
                text, phase = assistant_message
                return [{"kind": "agent_text", "text": text, "phase": phase}]
    if etype in ("item.completed", "item.started"):
        item = ev.get("item") or {}
        if item.get("type") == "agent_message":
            if etype == "item.completed":
                return [
                    {
                        "kind": "agent_text",
                        "text": item.get("text", ""),
                        "phase": item.get("phase", ""),
                    }
                ]
            return []
        if etype == "item.completed":
            return [{"kind": "progress", "text": _describe_item(item)}]
    if etype == "error":
        message = str(ev.get("message") or ev.get("error") or "error")
        return [{"kind": "progress", "text": f"warning: {message}"}]
    return []


def _codex_stderr_for_error(stderr_text: str, *, returncode: int | None, has_final_text: bool) -> str:
    """Drop known Codex CLI bookkeeping noise after successful calls."""
    if returncode not in (0, None) or not has_final_text:
        return stderr_text
    ignored_patterns = (
        "Reading prompt from stdin",
        "failed to record rollout items: thread",
    )
    kept_lines = [
        line
        for line in stderr_text.splitlines()
        if not any(pattern in line for pattern in ignored_patterns)
    ]
    return "\n".join(kept_lines).strip()
