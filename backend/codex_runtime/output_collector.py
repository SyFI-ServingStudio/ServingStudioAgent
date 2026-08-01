"""Collect Codex stdout/stderr/rollout output into UI-facing events."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from ..logging_config import compact_text, log_event
from .codex_events import (
    _codex_stderr_for_error,
    _find_rollout_file,
    _scan_rollout_agent_messages,
    _scan_rollout_last_token_usage,
    _translate,
    parse_commentary,
)
from .config import CODEX_IDLE_TIMEOUT, LOG
from .exec_types import CodexEvent, CodexExecRequest


class CodexOutputCollector:
    """Owns the stateful interpretation of one Codex subprocess call."""

    def __init__(self, request: CodexExecRequest) -> None:
        self.request = request
        self.current_session_id = request.session_id
        self.final_text: str | None = None
        self.timed_out = False
        self.rollout_file: Path | None = None
        self.rollout_offset = 0
        self.stderr_chunks: list[str] = []
        self.seen_intermediate_outputs: set[tuple[str, str, str]] = set()
        # Cumulative token usage recorded just before this call, so the per-call
        # delta is `end - baseline` (the Codex session is reused across rounds).
        self.tokens_baseline: dict[str, int] | None = None

    def prime_rollout_offset(self) -> None:
        if not self.current_session_id:
            return
        self.rollout_file = _find_rollout_file(
            self.request.workspace_id,
            self.request.conversation_id,
            self.request.label,
            self.current_session_id,
        )
        if self.rollout_file is not None:
            with contextlib.suppress(OSError):
                self.rollout_offset = self.rollout_file.stat().st_size
            self.tokens_baseline = _scan_rollout_last_token_usage(self.rollout_file)

    def append_stderr_line(self, line: bytes) -> None:
        self.stderr_chunks.append(line.decode("utf-8", "replace"))

    def mark_timed_out(self) -> None:
        self.timed_out = True

    def events_from_stdout_line(self, raw_line: bytes) -> list[CodexEvent]:
        events: list[CodexEvent] = []
        for translated_event in _translate_stdout_line(raw_line):
            event = self._event_from_translated_event(translated_event)
            if event is not None:
                events.append(event)
        return events

    def poll_rollout_intermediate_outputs(self) -> list[CodexEvent]:
        if not self.current_session_id:
            return []
        if self.rollout_file is None:
            self.rollout_file = _find_rollout_file(
                self.request.workspace_id,
                self.request.conversation_id,
                self.request.label,
                self.current_session_id,
            )
            if self.rollout_file is None:
                return []

        messages, self.rollout_offset = _scan_rollout_agent_messages(
            self.rollout_file,
            self.rollout_offset,
        )
        events = []
        for text, phase in messages:
            if phase != "commentary":
                continue
            event = self._intermediate_output_event(text.strip(), source="rollout")
            if event is not None:
                events.append(event)
        return events

    def timeout_event(self) -> CodexEvent:
        log_event(
            LOG,
            "codex.timeout",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.label,
            idle_timeout_s=CODEX_IDLE_TIMEOUT,
        )
        return {
            "kind": "error",
            "text": (
                f"{self.request.label}: codex call timed out after "
                f"{CODEX_IDLE_TIMEOUT:.0f}s without output"
            ),
        }

    def usage_event(self, duration_ms: int) -> CodexEvent:
        """Per-call wall-clock + token breakdown (prefix read / prefill / output)."""
        tokens = self._token_delta()
        log_event(
            LOG,
            "codex.usage",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.label,
            model=self.request.model_id,
            effort=self.request.effort,
            duration_ms=duration_ms,
            read_tokens=tokens["read"],
            prefill_tokens=tokens["prefill"],
            output_tokens=tokens["output"],
        )
        return {
            "kind": "usage",
            "role": self.request.label,
            "model": self.request.model_id,
            "effort": self.request.effort,
            "duration_ms": duration_ms,
            "tokens": tokens,
        }

    def _token_delta(self) -> dict[str, int]:
        """`end - baseline` cumulative usage, split into UI token buckets.

        prefix read = cached input; prefill (cache write) = fresh input;
        output = generated tokens. Missing rollout / negative deltas clamp to 0.
        """
        if self.rollout_file is None and self.current_session_id:
            self.rollout_file = _find_rollout_file(
                self.request.workspace_id,
                self.request.conversation_id,
                self.request.label,
                self.current_session_id,
            )
        end = (
            _scan_rollout_last_token_usage(self.rollout_file)
            if self.rollout_file is not None
            else None
        ) or {}
        base = self.tokens_baseline or {}

        def field(usage: dict[str, int], key: str) -> int:
            value = usage.get(key)
            return int(value) if isinstance(value, (int, float)) else 0

        read = field(end, "cached_input_tokens") - field(base, "cached_input_tokens")
        prefill = (field(end, "input_tokens") - field(end, "cached_input_tokens")) - (
            field(base, "input_tokens") - field(base, "cached_input_tokens")
        )
        output = field(end, "output_tokens") - field(base, "output_tokens")
        return {
            "read": max(0, read),
            "prefill": max(0, prefill),
            "output": max(0, output),
        }

    def final_event(self, returncode: int | None) -> CodexEvent:
        raw_stderr_text = "".join(self.stderr_chunks).strip()
        stderr_text = _codex_stderr_for_error(
            raw_stderr_text,
            returncode=returncode,
            has_final_text=self.final_text is not None,
        )
        final_text = self.final_text or self._fallback_final_text(
            returncode, stderr_text
        )
        log_event(
            LOG,
            "codex.final",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.label,
            returncode=returncode,
            final_len=len(final_text),
            final_preview=compact_text(final_text),
            stderr_len=len(stderr_text),
            stderr_tail=stderr_text[-500:] if stderr_text else "",
            raw_stderr_len=len(raw_stderr_text),
        )
        return {"kind": "final", "text": final_text}

    def _event_from_translated_event(
        self, translated_event: CodexEvent
    ) -> CodexEvent | None:
        kind = translated_event["kind"]
        if kind == "session":
            return self._session_event(translated_event["session_id"])
        if kind == "agent_text":
            return self._capture_agent_text(translated_event)
        return self._tool_call_event(translated_event.get("text", ""))

    def _session_event(self, session_id: str) -> CodexEvent:
        self.current_session_id = session_id
        log_event(
            LOG,
            "codex.session",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.label,
            model=self.request.model_id,
            effort=self.request.effort,
            codex_session_id=session_id,
        )
        return {
            "kind": "session",
            "role": self.request.label,
            "model": self.request.model_id,
            "effort": self.request.effort,
            "session_id": session_id,
        }

    def _capture_agent_text(self, translated_event: CodexEvent) -> CodexEvent | None:
        text = translated_event.get("text") or ""
        phase = translated_event.get("phase") or ""
        if phase == "commentary":
            return self._intermediate_output_event(text.strip(), source="stdout")
        if text.strip():
            self.final_text = text.strip()
        return None

    def _intermediate_output_event(
        self, note_text: str, *, source: str
    ) -> CodexEvent | None:
        note_text, level = parse_commentary(note_text)
        if not note_text:
            return None
        # The same words may intentionally be promoted from a small progress
        # note to a completed milestone. Keep the semantic level in the key.
        note_key = (self.request.label, level, note_text)
        if note_key in self.seen_intermediate_outputs:
            return None
        self.seen_intermediate_outputs.add(note_key)
        log_event(
            LOG,
            "codex.intermediate_output",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.label,
            model=self.request.model_id,
            effort=self.request.effort,
            text=note_text,
            source=source,
        )
        return {
            "kind": "intermediate_output",
            "role": self.request.label,
            "model": self.request.model_id,
            "effort": self.request.effort,
            "level": level,
            "text": note_text,
        }

    def _tool_call_event(self, tool_call_text: str) -> CodexEvent:
        log_event(
            LOG,
            "codex.tool_call",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.label,
            text=tool_call_text,
        )
        return {
            "kind": "tool_call",
            "text": f"{self.request.label}: {tool_call_text}",
        }

    def _fallback_final_text(self, returncode: int | None, stderr_text: str) -> str:
        if returncode not in (0, None) and stderr_text:
            return f"({self.request.label} exited {returncode})\n\n```\n{stderr_text[-1500:]}\n```"
        if stderr_text:
            return f"({self.request.label} produced no final text)\n\n```\n{stderr_text[-1500:]}\n```"
        return f"({self.request.label} produced no final text)"


def _translate_stdout_line(raw_line: bytes) -> list[CodexEvent]:
    line = raw_line.decode("utf-8", "replace").strip()
    if not line:
        return []
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return []
    return _translate(event)
