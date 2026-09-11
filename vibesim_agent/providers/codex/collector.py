"""Collect Codex stdout/stderr/rollout output into UI-facing events."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from .events import (
    _codex_stderr_for_error,
    _find_rollout_file,
    _scan_rollout_agent_messages,
    _scan_rollout_last_token_usage,
    _translate,
    parse_commentary,
    terminal_envelope,
    transport_failure,
)
from ..base import AgentRequest

CodexEvent = dict[str, Any]


class CodexOutputCollector:
    """Owns the stateful interpretation of one Codex subprocess call."""

    def __init__(self, request: AgentRequest, *, home: Path, idle_timeout: float, logger: logging.Logger) -> None:
        self.request = request
        self.home = home
        self.idle_timeout = idle_timeout
        self.logger = logger
        self.current_session_id = request.session_id
        self.ready = False
        self.final_text: str | None = None
        self.timed_out = False
        self.rollout_file: Path | None = None
        self.rollout_offset = 0
        self.stderr_tail = ""
        self.stderr_chars_seen = 0
        self.stderr_failure: dict[str, Any] | None = None
        self.seen_intermediate_outputs: set[tuple[str, str, str]] = set()
        # Some Responses-compatible providers omit Codex's commentary phase.
        # Keep their assistant text pending until the next event establishes
        # whether work continued (intermediate) or the call ended (final).
        self.pending_unphased_text: str | None = None
        # Rollout terminal events are authoritative when stdout ordering makes
        # a completed handoff look like an intermediate message.
        self.rollout_terminal_text: str | None = None
        # Cumulative token usage recorded just before this call, so the per-call
        # delta is `end - baseline` (the Codex session is reused across rounds).
        self.tokens_baseline: dict[str, int] | None = None
        # The upstream status last seen on this call, if any. Reconnect notices
        # set it too, so it only means "this call failed" once the call has also
        # ended without assistant text — see `final_event`.
        self.transport_failure: dict[str, Any] | None = None

    def _log(self, event: str, **fields: Any) -> None:
        self.logger.info(event, extra={"event_fields": {"event": event, **fields}})

    def prime_rollout_offset(self) -> None:
        if not self.current_session_id:
            return
        self.rollout_file = _find_rollout_file(
            self.home,
            self.current_session_id,
        )
        if self.rollout_file is not None:
            with contextlib.suppress(OSError):
                self.rollout_offset = self.rollout_file.stat().st_size
            self.tokens_baseline = _scan_rollout_last_token_usage(self.rollout_file)

    def append_stderr_line(self, line: bytes) -> None:
        text = line.decode("utf-8", "replace")
        self.stderr_chars_seen += len(text)
        combined = self.stderr_tail + text
        # Preserve the first stderr status even after later noise evicts its text.
        if self.stderr_failure is None:
            self.stderr_failure = transport_failure(combined)
        self.stderr_tail = combined[-8192:]

    def mark_timed_out(self) -> None:
        self.timed_out = True

    def events_from_stdout_line(self, raw_line: bytes) -> list[CodexEvent]:
        events: list[CodexEvent] = []
        for translated_event in _translate_stdout_line(raw_line):
            if translated_event["kind"] == "tool_call":
                pending_event = self._flush_pending_unphased_text()
                if pending_event is not None:
                    events.append(pending_event)
            event = self._event_from_translated_event(translated_event)
            if event is not None:
                events.append(event)
        return events

    def poll_rollout_intermediate_outputs(self) -> list[CodexEvent]:
        if not self.current_session_id:
            return []
        if self.rollout_file is None:
            self.rollout_file = _find_rollout_file(
                self.home,
                self.current_session_id,
            )
            if self.rollout_file is None:
                return []

        messages, terminal_text, self.rollout_offset = _scan_rollout_agent_messages(
            self.rollout_file,
            self.rollout_offset,
        )
        if terminal_text:
            self.rollout_terminal_text = terminal_text
            self.ready = True
        events = []
        for text, phase in messages:
            if text.strip():
                self.ready = True
            if phase != "commentary":
                continue
            event = self._intermediate_output_event(text.strip(), source="rollout")
            if event is not None:
                events.append(event)
        return events

    def timeout_event(self) -> CodexEvent:
        # An idle timeout is the same class of failure as an upstream outage:
        # the call ends with no decision to parse. Marking it here stops the
        # turn loop from spending its repair rounds — three more idle timeouts —
        # on a call that never reached the model.
        self.transport_failure = {"code": "codex_call_timeout", "status": 0}
        self._log(
            "codex.timeout",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.role.value,
            idle_timeout_s=self.idle_timeout,
        )
        return {
            "kind": "error",
            "text": (
                f"{self.request.role.value}: codex call timed out after "
                f"{self.idle_timeout:.0f}s without output"
            ),
        }

    def usage_event(self, duration_ms: int) -> CodexEvent:
        """Per-call wall-clock + token breakdown (prefix read / prefill / output)."""
        tokens = self._token_delta()
        self._log(
            "codex.usage",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.role.value,
            model=self.request.selection.model.model_id,
            effort=self.request.selection.effort,
            duration_ms=duration_ms,
            read_tokens=tokens["read"],
            prefill_tokens=tokens["prefill"],
            output_tokens=tokens["output"],
        )
        return {
            "kind": "usage",
            "role": self.request.role.value,
            "model": self.request.selection.model.model_id,
            "effort": self.request.selection.effort,
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
                self.home,
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
        if self.rollout_terminal_text:
            self.final_text = self.rollout_terminal_text
            self.pending_unphased_text = None
        elif self.pending_unphased_text:
            self.final_text = self.pending_unphased_text
            self.pending_unphased_text = None
        raw_stderr_text = self.stderr_tail.strip()
        stderr_text = _codex_stderr_for_error(
            raw_stderr_text,
            returncode=returncode,
            has_final_text=self.final_text is not None,
        )
        # A status seen mid-call means nothing if the call still produced an
        # answer — the CLI reconnected. Only a call that ended empty is a
        # transport failure, and only then is stderr worth scanning: on a
        # successful call any status text there is incidental (a sandboxed
        # command's own output), not the cause.
        failure = self.transport_failure if self.final_text is None else None
        if failure is None and self.final_text is None:
            failure = self.stderr_failure
        final_text = self.final_text or self._fallback_final_text(
            returncode, stderr_text, failure
        )
        self._log(
            "codex.final",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.role.value,
            returncode=returncode,
            final_len=len(final_text),
            final_preview=" ".join(final_text.split())[:240],
            stderr_len=len(stderr_text),
            stderr_tail=stderr_text[-500:] if stderr_text else "",
            raw_stderr_len=self.stderr_chars_seen,
            failure_code=failure["code"] if failure else "",
            failure_status=failure["status"] if failure else 0,
        )
        event: CodexEvent = {"kind": "final", "text": final_text}
        if failure is not None:
            event["failure"] = dict(failure)
        return event

    def _event_from_translated_event(
        self, translated_event: CodexEvent
    ) -> CodexEvent | None:
        kind = translated_event["kind"]
        if kind == "session":
            return self._session_event(translated_event["session_id"])
        if kind == "agent_text":
            if (translated_event.get("text") or "").strip():
                self.ready = True
            return self._capture_agent_text(translated_event)
        failure = translated_event.get("transport_failure")
        if failure:
            self.transport_failure = failure
        text = translated_event.get("text", "")
        if text and not text.startswith(("warning:", "note:")):
            self.ready = True
        return self._tool_call_event(text)

    def _session_event(self, session_id: str) -> CodexEvent:
        if session_id != self.current_session_id:
            self.rollout_file = None
            self.rollout_offset = 0
            self.tokens_baseline = None
            self.rollout_terminal_text = None
        self.current_session_id = session_id
        self._log(
            "codex.session",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.role.value,
            model=self.request.selection.model.model_id,
            effort=self.request.selection.effort,
            codex_session_id=session_id,
        )
        return {
            "kind": "session",
            "role": self.request.role.value,
            "model": self.request.selection.model.model_id,
            "effort": self.request.selection.effort,
            "session_id": session_id,
        }

    def _capture_agent_text(self, translated_event: CodexEvent) -> CodexEvent | None:
        text = translated_event.get("text") or ""
        phase = translated_event.get("phase") or ""
        if phase == "commentary":
            return self._intermediate_output_event(text.strip(), source="stdout")
        if phase == "final_answer" or terminal_envelope(text):
            if text.strip():
                self.final_text = text.strip()
            return None
        if text.strip():
            self.pending_unphased_text = text.strip()
        return None

    def _flush_pending_unphased_text(self) -> CodexEvent | None:
        """Promote pending assistant text once later tool activity proves it non-final."""
        if not self.pending_unphased_text:
            return None
        note_text = self.pending_unphased_text
        self.pending_unphased_text = None
        return self._intermediate_output_event(note_text, source="stdout-unphased")

    def _intermediate_output_event(
        self, note_text: str, *, source: str
    ) -> CodexEvent | None:
        note_text, level = parse_commentary(note_text)
        if not note_text:
            return None
        # The same words may intentionally be promoted from a small progress
        # note to a completed milestone. Keep the semantic level in the key.
        note_key = (self.request.role.value, level, note_text)
        if note_key in self.seen_intermediate_outputs:
            return None
        self.seen_intermediate_outputs.add(note_key)
        self._log(
            "codex.intermediate_output",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.role.value,
            model=self.request.selection.model.model_id,
            effort=self.request.selection.effort,
            text=note_text,
            source=source,
        )
        return {
            "kind": "intermediate_output",
            "role": self.request.role.value,
            "model": self.request.selection.model.model_id,
            "effort": self.request.selection.effort,
            "level": level,
            "text": note_text,
        }

    def _tool_call_event(self, tool_call_text: str) -> CodexEvent:
        self._log(
            "codex.tool_call",
            conversation_id=self.request.conversation_id,
            turn_id=self.request.turn_id,
            role=self.request.role.value,
            text=tool_call_text,
        )
        return {
            "kind": "tool_call",
            "text": f"{self.request.role.value}: {tool_call_text}",
        }

    def _fallback_final_text(
        self,
        returncode: int | None,
        stderr_text: str,
        failure: dict[str, Any] | None = None,
    ) -> str:
        # A known cause is the whole explanation, so state it instead of echoing
        # stderr — that text carries the gateway URL and request id.
        if failure is not None:
            if failure["status"]:
                return (
                    f"({self.request.role.value} produced no output: "
                    f"upstream returned {failure['status']})"
                )
            return (
                f"({self.request.role.value} produced no output before the "
                f"{self.idle_timeout:.0f}s idle timeout)"
            )
        if returncode not in (0, None) and stderr_text:
            return f"({self.request.role.value} exited {returncode})\n\n```\n{stderr_text[-1500:]}\n```"
        if stderr_text:
            return f"({self.request.role.value} produced no final text)\n\n```\n{stderr_text[-1500:]}\n```"
        return f"({self.request.role.value} produced no final text)"


def _translate_stdout_line(raw_line: bytes) -> list[CodexEvent]:
    line = raw_line.decode("utf-8", "replace").strip()
    if not line:
        return []
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return []
    return _translate(event)
