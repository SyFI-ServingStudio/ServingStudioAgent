"""Role prompts and orchestrator JSON parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from .config import PROMPTS_DIR


def _role_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _orchestrator_contract(conversation_id: str) -> str:
    return (
        f"{_role_prompt('orchestrator.txt')}\n\n"
        f"Current conversation ID: `{conversation_id}`.\n"
        f"Conversation plan: `/workspace/{conversation_id}_plan.md`.\n"
        f"Conversation progress: `/workspace/{conversation_id}_progress.md`."
    )


def _orchestrator_prompt(
    user_text: str,
    *,
    is_resume: bool,
    conversation_id: str,
) -> str:
    # The resumed session supplies history; this prefix supplies the latest
    # repository role contract without invalidating that history.
    del is_resume
    return (
        f"{_orchestrator_contract(conversation_id)}"
        f"\n\nNewest user message:\n{user_text}\n"
    )


def _orchestrator_handoff_prompt(
    task: str,
    implementer_text: str,
    *,
    conversation_id: str,
) -> str:
    return (
        f"{_orchestrator_contract(conversation_id)}\n\n"
        "The implementer returned a summary for your delegated task.\n\n"
        "You do not share the implementer Codex session. Treat the text below as "
        "the explicit handoff record, review it against your own orchestration "
        "context, and return exactly one JSON object.\n\n"
        "If the work is complete, use `final_answer`. If clarification, "
        "authorization, or an external choice is genuinely required, use "
        "`request_user_input`. If another bounded code-change, validation, or "
        "large exploration task is still needed, use `delegate` with that "
        "specific follow-up task.\n\n"
        "Delegated task:\n"
        f"{task}\n\n"
        "Implementer summary:\n"
        f"{implementer_text}\n"
    )


def _orchestrator_repair_prompt(
    unparsed_output: str,
    *,
    conversation_id: str,
) -> str:
    """Ask a resumed orchestrator to repair only its decision envelope."""
    return (
        f"{_orchestrator_contract(conversation_id)}\n\n"
        "Your previous final output could not be parsed as the required decision "
        "JSON. Do not redo completed analysis. Return exactly one JSON object with "
        "the fields `action`, `message`, and `task`. If the text below is the "
        "completed answer, preserve it in `message` with action `final_answer`. "
        "If user input is genuinely required, use `request_user_input`; if a "
        "separate implementer task is required, use `delegate`. Do not end with "
        "a progress update and do not add text outside the JSON object.\n\n"
        f"Unparsed previous output:\n{unparsed_output}\n"
    )


def _orchestrator_continue_prompt(
    action: str,
    message: str,
    *,
    conversation_id: str,
) -> str:
    """Resume after a non-terminal envelope was emitted as the final item."""
    return (
        f"{_orchestrator_contract(conversation_id)}\n\n"
        f"Your previous call ended with a non-terminal `{action}` update. The "
        "runtime already showed it to the user. Continue the same work from that "
        "checkpoint without repeating completed analysis. End this call only with "
        "`final_answer`, `request_user_input`, or `delegate`; use `progress` and "
        "`milestone` only for commentary emitted while you keep working.\n\n"
        f"Last update:\n{message}\n"
    )


def _implementer_prompt(
    task: str,
    *,
    is_resume: bool,
) -> str:
    # Keep the role contract explicit on every Codex call. A resumed session
    # has prior context, but the new delegated task must still be framed as
    # implementor work rather than relying on that context implicitly.
    del is_resume
    return f"{_role_prompt('implementer.txt')}\n\nTask:\n{task}\n"


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


def parse_orchestrator(text: str) -> dict[str, Any] | None:
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        action = payload.get("action")
        message = payload.get("message")
        task = payload.get("task")
        message_is_empty = message is None or (
            isinstance(message, str) and not message.strip()
        )
        task_is_empty = task is None or (
            isinstance(task, str) and not task.strip()
        )
        if action in {"progress", "milestone"} and isinstance(
            message, str
        ) and message.strip() and task_is_empty:
            return {
                "action": action,
                "message": _normalize_orchestrator_text_field(message),
            }
        if action in {"delegate", "run_implementer"} and isinstance(
            task, str
        ) and task.strip() and message_is_empty:
            return {
                "action": "delegate",
                "task": _normalize_orchestrator_text_field(task),
            }
        if action in {"final_answer", "respond", "user_message"} and isinstance(
            message, str
        ) and message.strip() and task_is_empty:
            return {
                "action": "final_answer",
                "message": _normalize_orchestrator_text_field(message),
            }
        if action == "request_user_input" and isinstance(
            message, str
        ) and message.strip() and task_is_empty:
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


def compose_final_message(
    message: str,
    implementer_summaries: list[str],
) -> str:
    """The orchestrator's final answer, without the implementer summaries.

    Implementer conclusions render as their own timeline cards (and stay
    available separately as `implementer_summaries` on the JSON turn result), so
    embedding them here would duplicate the same text inside the answer card.
    `implementer_summaries` is kept in the signature for call-site compatibility.
    """
    return message.strip()


def compose_failure_message(reason: str, implementer_summaries: list[str]) -> str:
    """A failed turn's body: reports completed work by count, never by quoting it.

    Same contract as `compose_final_message`. Quoting the summaries here would
    both duplicate their timeline cards and attribute the implementer's
    first-person report to the assistant's own answer, which reads as the two
    roles having been confused.
    """
    rounds = len(implementer_summaries)
    if not rounds:
        return reason.strip()
    noun = "round" if rounds == 1 else "rounds"
    return (
        f"{reason.strip()}\n\n"
        f"{rounds} implementer {noun} completed before this failure and are shown "
        "as their own cards above. The Codex sessions are preserved — continue "
        "the conversation to resume from this point."
    )


def transport_failure_reason(role: str, failure: dict[str, Any]) -> str:
    """Plain-language cause for one failed Codex call.

    Names the transport as the cause so an outage is not read as a model or
    parsing problem, which is what the generic wording used to imply.
    """
    if failure.get("code") == "codex_call_timeout":
        return (
            f"The {role} produced no output before its idle timeout, so this "
            "turn has no answer. The call reached no decision — this is a "
            "runtime stall, not a model or parsing problem."
        )
    status = failure.get("status") or 0
    if failure.get("code") == "upstream_rate_limited":
        return (
            f"The {role} could not run: the upstream model gateway is rate "
            f"limiting this account (HTTP {status}), and the Codex CLI already "
            "exhausted its own retries. No output was produced — this is an API "
            "limit, not a model or parsing problem."
        )
    return (
        f"The {role} could not run: the upstream model gateway is unavailable "
        f"(HTTP {status}), and the Codex CLI already exhausted its own "
        "reconnects. No output was produced — this is an API outage, not a "
        "model or parsing problem."
    )
