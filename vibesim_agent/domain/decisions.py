"""Pure role decision parsing shared by turn entry points."""

import json
import re
from typing import Any

# Envelopes a role sends while it keeps working. They never end a turn: the
# driver shows one and resumes the role if a call stops on it.
COMMENTARY_ACTIONS = frozenset({"progress", "milestone"})


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped] if stripped else []
    candidates.extend(
        m.strip()
        for m in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    )
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    return candidates


def _normalize_orchestrator_text_field(value: str) -> str:
    """Make text fields readable when the model double-escapes JSON newlines."""
    return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def parse_orchestrator(
    text: str,
    *,
    allow_delegate: bool = True,
) -> dict[str, Any] | None:
    """Parse one decision envelope.

    `allow_delegate=False` is the single-agent contract: a `delegate` payload is
    rejected rather than normalized, so it falls through to `None` and the
    caller spends a repair round telling the model to finish the work itself.
    """
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        action = payload.get("action")
        if action is not None and not isinstance(action, str):
            continue
        message = payload.get("message")
        task = payload.get("task")
        message_is_empty = message is None or (
            isinstance(message, str) and not message.strip()
        )
        task_is_empty = task is None or (isinstance(task, str) and not task.strip())
        if (
            action in COMMENTARY_ACTIONS
            and isinstance(message, str)
            and message.strip()
            and task_is_empty
        ):
            return {
                "action": action,
                "message": _normalize_orchestrator_text_field(message),
            }
        if action in {"delegate", "run_implementer"} and not allow_delegate:
            continue
        if (
            action in {"delegate", "run_implementer"}
            and isinstance(task, str)
            and task.strip()
            and message_is_empty
        ):
            return {
                "action": "delegate",
                "task": _normalize_orchestrator_text_field(task),
            }
        if (
            action in {"final_answer", "respond", "user_message"}
            and isinstance(message, str)
            and message.strip()
            and task_is_empty
        ):
            return {
                "action": "final_answer",
                "message": _normalize_orchestrator_text_field(message),
            }
        if (
            action == "request_user_input"
            and isinstance(message, str)
            and message.strip()
            and task_is_empty
        ):
            return {
                "action": "request_user_input",
                "message": _normalize_orchestrator_text_field(message),
            }
        if action is None and isinstance(payload.get("message"), str):
            return {
                "action": "final_answer",
                "message": _normalize_orchestrator_text_field(payload["message"]),
            }
        if action is None and payload:
            return {
                "action": "final_answer",
                "message": json.dumps(payload, ensure_ascii=False, indent=2),
            }
    return None


def parse_implementer(
    text: str,
    *,
    allow_reply_user: bool,
) -> dict[str, str]:
    """Read one implementer result envelope, falling back to the raw text.

    An unparseable answer is not repaired the way the driver's is: the work it
    describes is already done, and the summary is text either way. So anything
    that is not a recognizable envelope becomes a `final_answer` carrying the
    model's own words, which is exactly the pre-envelope behaviour.

    `allow_reply_user=False` is the delegated path, where the prompt carried no
    `user:` line. A `reply_user` there would end the turn on an answer to a
    question nobody asked, so it is demoted rather than obeyed — the
    orchestrator still gets its summary and decides how the turn ends.

    A `progress` or `milestone` envelope is returned as itself. A call that
    stops on one has not finished its task, and handing that update to the
    orchestrator as the summary would review half-done work as done.
    """
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        action = payload.get("action")
        if action is not None and not isinstance(action, str):
            continue
        if action not in {"final_answer", "reply_user", *COMMENTARY_ACTIONS}:
            continue
        if action == "reply_user" and not allow_reply_user:
            action = "final_answer"
        return {
            "action": action,
            "message": _normalize_orchestrator_text_field(message),
        }
    return {"action": "final_answer", "message": text}
