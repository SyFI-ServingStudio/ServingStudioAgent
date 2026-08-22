"""Fold a `run_turn` event stream into one JSON-friendly result dict.

`run_turn` (backend/codex_runtime/turn.py) is an async generator that yields
dict events with a `kind` of session/tool_call/error/intermediate_output/
decision/usage/implementer/final. Both JSON entry points — `/api/eval`
(backend/eval.py) and the
agent conversation turn (backend/app.py) — drain that stream into the same shape
using `new_turn_result` + `collect_turn_event`. (The browser path streams the
events as SSE instead and does not collect them here.)
"""

from __future__ import annotations

from typing import Any

from .codex_runtime.config import DEFAULT_AGENT_MODE


def new_turn_result(
    *,
    conversation_id: str,
    turn_id: str,
    sandbox: str,
    autonomous: bool,
    agent_mode: str = DEFAULT_AGENT_MODE,
    **extra: Any,
) -> dict[str, Any]:
    """Build the base result dict shared by the eval and agent-turn endpoints.

    `extra` carries endpoint-specific fields (e.g. eval's workspace/kept_* flags).
    `implementer_summaries` / `delegated_tasks` stay in the shape under
    `agent_mode="single"` — the single agent never delegates, so they stay empty
    rather than disappearing from a contract callers already depend on.
    """
    result: dict[str, Any] = {
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "sandbox": sandbox,
        "autonomous": autonomous,
        "agent_mode": agent_mode,
        "sessions": {},
        "tool_calls": [],
        "intermediate_outputs": [],
        "implementer_summaries": [],
        "delegated_tasks": [],
        "usages": [],
        "final": "",
        "outcome": None,
        "ok": False,
        "error": "",
        "failure_code": "",
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
    elif kind == "tool_call":
        result["tool_calls"].append(str(event.get("text") or ""))
    elif kind == "error":
        result["tool_calls"].append(str(event.get("text") or ""))
        result["error"] = str(event.get("text") or "")
    elif kind == "intermediate_output":
        result["intermediate_outputs"].append(
            {
                "role": str(event.get("role") or ""),
                "model": str(event.get("model") or ""),
                "effort": str(event.get("effort") or ""),
                "level": (
                    "milestone"
                    if event.get("level") == "milestone"
                    else "progress"
                ),
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
        failure = event.get("failure")
        if isinstance(failure, dict):
            # The turn ended without an orchestrator decision, so it has no
            # outcome. `failure_code` is what the caller maps to the same
            # {code, message} contract the browser path publishes.
            result["failure_code"] = str(failure.get("code") or "")
            result["outcome"] = None
            return
        outcome = event.get("outcome")
        result["outcome"] = (
            outcome
            if outcome in {"final_answer", "request_user_input"}
            else "final_answer"
        )
