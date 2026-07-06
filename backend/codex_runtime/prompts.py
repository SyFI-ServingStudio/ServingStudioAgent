"""Role prompts and orchestrator JSON parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from .config import PROMPTS_DIR

def _role_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()

def _orchestrator_prompt(user_text: str, *, is_resume: bool) -> str:
    if is_resume:
        return user_text
    return f"{_role_prompt('orchestrator.txt')}\n\nNewest user message:\n{user_text}\n"

def _orchestrator_handoff_prompt(task: str, implementer_text: str) -> str:
    return (
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

def _implementer_prompt(task: str, *, is_resume: bool) -> str:
    if is_resume:
        return f"Task:\n{task}\n"
    return f"{_role_prompt('implementer.txt')}\n\nTask:\n{task}\n"

def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped] if stripped else []
    candidates.extend(m.strip() for m in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    return candidates

def _normalize_orchestrator_text_field(value: str) -> str:
    """Make text fields readable when the model double-escapes JSON newlines."""
    return (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
    )

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
    if not implementer_summaries:
        return message.strip()
    sections = []
    if implementer_summaries:
        sections.append(
            "### Implementer Summary\n\n"
            f"{_format_implementer_summaries(implementer_summaries)}"
        )
    sections.append(f"### Message\n\n{message.strip()}")
    return "\n\n".join(sections)
