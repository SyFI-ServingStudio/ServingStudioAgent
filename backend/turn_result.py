"""Fold a `run_turn` event stream into one JSON-friendly result dict.

`run_turn` (backend/codex_runtime/turn.py) is an async generator that yields
dict events with a `kind` of session/progress/error/intermediate_output/
decision/usage/implementer/final. Both JSON entry points — `/api/eval`
(backend/eval.py) and the
agent conversation turn (backend/app.py) — drain that stream into the same shape
using `new_turn_result` + `collect_turn_event`. (The browser path streams the
events as SSE instead and does not collect them here.)
"""

from __future__ import annotations

from typing import Any


def new_turn_result(
    *,
    conversation_id: str,
    turn_id: str,
    sandbox: str,
    autonomous: bool,
    **extra: Any,
) -> dict[str, Any]:
    """Build the base result dict shared by the eval and agent-turn endpoints.

    `extra` carries endpoint-specific fields (e.g. eval's workspace/kept_* flags).
    """
    result: dict[str, Any] = {
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "sandbox": sandbox,
        "autonomous": autonomous,
        "sessions": {},
        "progress": [],
        "intermediate_outputs": [],
        "implementer_summaries": [],
        "delegated_tasks": [],
        "usages": [],
        "final": "",
        "ok": False,
        "error": "",
    }
    result.update(extra)
    return result


def collect_turn_event(result: dict[str, Any], event: dict[str, str]) -> None:
    """Accumulate one run_turn event into a result dict from `new_turn_result`."""
    kind = event.get("kind")
    if kind == "session":
        role = str(event.get("role") or "")
        session_id = str(event.get("session_id") or "")
        if role and session_id:
            result["sessions"][role] = session_id
    elif kind == "progress":
        result["progress"].append(str(event.get("text") or ""))
    elif kind == "error":
        result["progress"].append(str(event.get("text") or ""))
        result["error"] = str(event.get("text") or "")
    elif kind == "intermediate_output":
        result["intermediate_outputs"].append(
            {
                "role": str(event.get("role") or ""),
                "text": str(event.get("text") or ""),
            }
        )
    elif kind == "implementer":
        result["implementer_summaries"].append(str(event.get("text") or ""))
    elif kind == "decision":
        task = str(event.get("task") or "")
        if task:
            result["delegated_tasks"].append(task)
    elif kind == "usage":
        result["usages"].append(
            {
                "role": str(event.get("role") or ""),
                "duration_ms": int(event.get("duration_ms") or 0),
                "tokens": event.get("tokens") or {},
            }
        )
    elif kind == "final":
        result["final"] = str(event.get("text") or "")
