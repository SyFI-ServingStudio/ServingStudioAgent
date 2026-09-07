"""Translate Claude stream-json into the existing browser event contract."""

import json

from jsonschema import Draft202012Validator

from .claude_command import output_schema
from .exec_types import CodexEvent, CodexExecRequest


class ClaudeOutputCollector:
    def __init__(self, request: CodexExecRequest):
        self.request = request
        schema = output_schema(request)
        self.validator = Draft202012Validator(schema) if schema else None
        self.result: dict | None = None
        self.session_id = request.session_id
        self.ready = False
        self.seen_messages: set[str] = set()

    def events_from_stdout_line(self, raw: bytes) -> list[CodexEvent]:
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return []
        if not isinstance(event, dict):
            return []
        events: list[CodexEvent] = []
        # Nested agent output must never replace the owning role's session or
        # become its routing decision.
        if event.get("parent_tool_use_id"):
            return events
        session = event.get("session_id")
        if isinstance(session, str) and session and session != self.session_id:
            self.session_id = session
            events.append(
                {"kind": "session", "role": self.request.label, "session_id": session}
            )
        kind = event.get("type")
        if kind in {"assistant", "user", "result"} and not self.ready:
            self.ready = True
            events.append({"kind": "role_ready", "role": self.request.label})
        if kind == "result":
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
            if identity and identity in self.seen_messages:
                return events
            if identity:
                self.seen_messages.add(identity)
            for block in message.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    events.append(
                        {
                            "kind": "tool_call",
                            "text": f"{self.request.label}: {block.get('name', 'tool')}",
                        }
                    )
                elif (
                    block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    and block["text"].strip()
                ):
                    text = block["text"].strip()
                    level = "progress"
                    try:
                        envelope = json.loads(text)
                    except ValueError:
                        envelope = None
                    if isinstance(envelope, dict):
                        if envelope.get("action") not in {"progress", "milestone"}:
                            continue  # Only the final result can route a turn.
                        level = envelope["action"]
                        text = envelope.get("message", "")
                    if isinstance(text, str) and text.strip():
                        events.append(
                            {
                                "kind": "intermediate_output",
                                "role": self.request.label,
                                "model": self.request.model_id,
                                "effort": self.request.effort,
                                "level": level,
                                "text": text,
                            }
                        )
        return events

    def finish(
        self, returncode: int, duration_ms: int, *, timed_out: bool = False
    ) -> list[CodexEvent]:
        result = self.result or {}
        usage = result.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}

        def tokens(key: str) -> int:
            value = usage.get(key, 0)
            return max(0, int(value)) if isinstance(value, (int, float)) else 0

        events: list[CodexEvent] = []
        if not self.ready:
            events.append({"kind": "role_ready", "role": self.request.label})
        events.append(
            {
                "kind": "usage",
                "role": self.request.label,
                "model": self.request.model_id,
                "effort": self.request.effort,
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
        final: CodexEvent = {"kind": "final", "text": text}
        if code:
            # Do not expose upstream error bodies, credentials, or tool output.
            final["failure"] = {"code": code, "status": 0}
        events.append(final)
        return events
