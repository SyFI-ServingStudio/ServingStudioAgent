"""Project durable turn events into the legacy synchronous response shape."""

from typing import Any

from .turns import Outcome, TurnInput, TurnResult


def project_turn_result(
    request: TurnInput, result: TurnResult, events: list[dict[str, Any]]
) -> dict[str, Any]:
    projected: dict[str, Any] = {
        "conversation_id": request.conversation_id,
        "turn_id": request.turn_id,
        "sandbox": request.sandbox.value,
        "autonomous": request.autonomous,
        "agent_mode": request.mode.value,
        "sessions": {},
        "tool_calls": [],
        "intermediate_outputs": [],
        "implementer_summaries": [],
        "delegated_tasks": [],
        "usages": [],
        "final": result.text,
        "outcome": None if result.outcome is Outcome.FAILED else result.outcome.value,
        "ok": False,
        "error": "",
        "failure_code": "",
    }
    has_error = False
    for row in events:
        event = row["payload"]
        kind = row["kind"]
        if kind == "session":
            role = str(event.get("role") or "")
            session_id = str(event.get("session_id") or "")
            if role and session_id:
                projected["sessions"][role] = session_id
        elif kind in {"tool_call", "error"}:
            text = str(event.get("text") or "")
            projected["tool_calls"].append(text)
            if kind == "error":
                has_error = True
                projected["error"] = text
        elif kind == "intermediate_output":
            projected["intermediate_outputs"].append(
                {
                    "role": str(event.get("role") or ""),
                    "model": str(event.get("model") or ""),
                    "effort": str(event.get("effort") or ""),
                    "level": "milestone"
                    if event.get("level") == "milestone"
                    else "progress",
                    "text": str(event.get("text") or ""),
                }
            )
        elif kind == "implementer":
            projected["implementer_summaries"].append(str(event.get("text") or ""))
        elif kind == "decision":
            task = str(event.get("task") or "")
            if task:
                projected["delegated_tasks"].append(task)
        elif kind == "usage":
            projected["usages"].append(
                {
                    "role": str(event.get("role") or ""),
                    "duration_ms": int(event.get("duration_ms") or 0),
                    "tokens": event.get("tokens") or {},
                }
            )
    failure = result.metadata.get("failure")
    if isinstance(failure, dict):
        code = failure.get("code")
        projected["failure_code"] = code if isinstance(code, str) else ""
    if result.outcome is Outcome.FAILED:
        message = failure.get("message") if isinstance(failure, dict) else None
        projected["error"] = (
            message
            if isinstance(message, str) and message.strip()
            else "The agent could not complete this turn."
        )
    projected["ok"] = (
        result.outcome in {Outcome.ANSWER, Outcome.INPUT}
        and bool(result.text.strip())
        and not has_error
        and failure is None
    )
    return projected
