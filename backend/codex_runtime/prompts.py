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
        "If the work is complete, risky, blocked, or needs a user choice, use "
        "`user_message`. If another bounded code-change, validation, or large "
        "exploration task is still needed, use `run_implementer` with that "
        "specific follow-up task.\n\n"
        "Delegated task:\n"
        f"{task}\n\n"
        "Implementer summary:\n"
        f"{implementer_text}\n"
    )


def _orchestrator_continue_prompt(
    progress: str,
    *,
    conversation_id: str,
) -> str:
    """Resume a non-terminal orchestrator checkpoint in the same session."""
    return (
        f"{_orchestrator_contract(conversation_id)}\n\n"
        "Your previous decision was `continue_work`, so the user-facing task is "
        "not complete. Resume the same task now from the durable recovery state. "
        "Do not merely repeat the progress update. Continue working until you can "
        "return a completed `user_message`, a real blocker, or a concrete "
        "`run_implementer` handoff.\n\n"
        f"Previous progress update:\n{progress}\n"
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
        "completed answer, preserve it in `message` with action `user_message`. "
        "If work remains, use `continue_work`; if delegation is required, use "
        "`run_implementer`. Do not add text outside the JSON object.\n\n"
        f"Unparsed previous output:\n{unparsed_output}\n"
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
        if action == "run_implementer" and isinstance(payload.get("task"), str):
            return {
                "action": action,
                "task": _normalize_orchestrator_text_field(payload["task"]),
            }
        if action == "user_message" and isinstance(payload.get("message"), str):
            return {
                "action": action,
                "message": _normalize_orchestrator_text_field(payload["message"]),
            }
        if action == "continue_work" and isinstance(payload.get("message"), str):
            return {
                "action": action,
                "message": _normalize_orchestrator_text_field(payload["message"]),
            }
        if action is None and isinstance(payload.get("message"), str):
            return {
                "action": "user_message",
                "message": _normalize_orchestrator_text_field(payload["message"]),
            }
        if action is None and payload:
            return {
                "action": "user_message",
                "message": json.dumps(payload, ensure_ascii=False, indent=2),
            }
    return None


def _format_implementer_summaries(summaries: list[str]) -> str:
    if not summaries:
        return ""
    if len(summaries) == 1:
        return summaries[0].strip()
    parts = []
    for idx, summary in enumerate(summaries, start=1):
        parts.append(f"**Round {idx}**\n\n{summary.strip()}")
    return "\n\n".join(parts)


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
