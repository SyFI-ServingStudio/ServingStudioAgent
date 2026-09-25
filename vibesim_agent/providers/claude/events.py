"""Translate Claude stream-json into the existing browser event contract."""

import json
import math

from jsonschema import Draft202012Validator

from ..base import AgentRequest

# Which input names what a tool call is doing, in order of preference. Bash
# carries a short `description` of its command; the rest name their target.
_TOOL_DETAIL_KEYS = ("description", "command", "file_path", "path", "pattern", "query", "url")
_TOOL_DETAIL_LIMIT = 100


def _tool_detail(tool_input: object) -> str:
    """One line saying what a tool call does, or nothing when it says nothing."""
    if not isinstance(tool_input, dict):
        return ""
    for key in _TOOL_DETAIL_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            line = " ".join(value.split())
            if len(line) > _TOOL_DETAIL_LIMIT:
                line = line[: _TOOL_DETAIL_LIMIT - 1].rstrip() + "…"
            return line
    return ""


class ClaudeOutputCollector:
    def __init__(self, request: AgentRequest, *, schema: dict | None = None):
        self.request = request
        if request.structured_output and schema is None:
            raise ValueError("structured output requires a schema")
        if not request.structured_output:
            schema = None
        self.validator = Draft202012Validator(schema) if schema else None
        self.result: dict | None = None
        self.session_id = request.session_id
        self.ready = False
        self.seen_messages: set[str] = set()

    def events_from_stdout_line(self, raw: bytes) -> list[dict]:
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return []
        if not isinstance(event, dict):
            return []
        events: list[dict] = []
        # Nested agent output must never replace the owning role's session or
        # become its routing decision.
        if event.get("parent_tool_use_id"):
            return events
        session = event.get("session_id")
        if isinstance(session, str) and session and session != self.session_id:
            self.session_id = session
            events.append(
                {
                    "kind": "session",
                    "role": self.request.role.value,
                    "session_id": session,
                }
            )
        kind = event.get("type")
        if kind == "result":
            self._mark_ready(events)
            self.result = event
        elif kind == "assistant":
            message = event.get("message") or {}
            if not isinstance(message, dict) or not isinstance(
                message.get("content", []), list
            ):
                return events
            # One message can contain several tool and text blocks. The CLI
            # may repeat an event on reconnect; deduplicate the whole event.
            identity = event.get("uuid")
            if isinstance(identity, str) and identity in self.seen_messages:
                return events
            if isinstance(identity, str) and identity:
                self.seen_messages.add(identity)
            for block in message.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    self._mark_ready(events)
                    # A role that works without narrating is otherwise a row of
                    # bare tool names; the detail says what each step runs.
                    text = f"{self.request.role.value}: {block.get('name', 'tool')}"
                    detail = _tool_detail(block.get("input"))
                    events.append(
                        {
                            "kind": "tool_call",
                            "text": f"{text} — {detail}" if detail else text,
                        }
                    )
                elif (
                    block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    and block["text"].strip()
                ):
                    self._mark_ready(events)
                    text = block["text"].strip()
                    level = "progress"
                    try:
                        envelope = json.loads(text)
                    except ValueError:
                        envelope = None
                    if isinstance(envelope, dict):
                        action = envelope.get("action")
                        if not isinstance(action, str) or action not in {
                            "progress",
                            "milestone",
                        }:
                            continue  # Only the final result can route a turn.
                        level = action
                        text = envelope.get("message", "")
                    if isinstance(text, str) and text.strip():
                        events.append(
                            {
                                "kind": "intermediate_output",
                                "role": self.request.role.value,
                                "model": self.request.selection.model.model_id,
                                "effort": self.request.selection.effort,
                                "level": level,
                                "text": text,
                            }
                        )
        return events

    def _mark_ready(self, events: list[dict]) -> None:
        if not self.ready:
            self.ready = True
            events.append({"kind": "role_ready", "role": self.request.role.value})

    def finish(
        self, returncode: int, duration_ms: int, *, timed_out: bool = False
    ) -> list[dict]:
        result = self.result or {}
        usage = result.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}

        def tokens(key: str) -> int:
            value = usage.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return 0
            if isinstance(value, float) and not math.isfinite(value):
                return 0
            return max(0, int(value))

        events: list[dict] = []
        events.append(
            {
                "kind": "usage",
                "role": self.request.role.value,
                "model": self.request.selection.model.model_id,
                "effort": self.request.selection.effort,
                "duration_ms": duration_ms,
                "tokens": {
                    "read": tokens("cache_read_input_tokens"),
                    "prefill": tokens("input_tokens")
                    + tokens("cache_creation_input_tokens"),
                    "output": tokens("output_tokens"),
                },
            }
        )
        code = ""
        text = ""
        if timed_out:
            code = "agent_call_timeout"
        elif (
            returncode != 0
            or not result
            or result.get("is_error")
            or result.get("subtype") != "success"
        ):
            code = "agent_runtime_failure"
        elif self.validator:
            structured = result.get("structured_output")
            if not self.validator.is_valid(structured):
                code = "agent_invalid_output"
            else:
                text = json.dumps(structured, ensure_ascii=False)
        elif isinstance(result.get("result"), str) and result["result"].strip():
            text = result["result"]
        else:
            code = "agent_invalid_output"
        final: dict = {"kind": "final", "text": text}
        if code:
            # Do not expose upstream error bodies, credentials, or tool output.
            final["failure"] = {"code": code, "status": 0}
        events.append(final)
        return events
