"""Role prompts and orchestrator JSON parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from .config import AGENT_MODE_ROLE_PROMPT, PROMPTS_DIR


def _role_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _driver_contract(role_prompt_name: str, conversation_id: str) -> str:
    """The startup contract for whichever role emits the decision envelope.

    Orchestrated turns pass `orchestrator.txt`, single-agent turns
    `assistant.txt`; the conversation-scoped plan/progress files are the same
    recovery state either way.
    """
    return (
        f"{_role_prompt(role_prompt_name)}\n\n"
        f"Current conversation ID: `{conversation_id}`.\n"
        f"Conversation plan: `/workspace/{conversation_id}_plan.md`.\n"
        f"Conversation progress: `/workspace/{conversation_id}_progress.md`."
    )


def _orchestrator_contract(conversation_id: str) -> str:
    return _driver_contract("orchestrator.txt", conversation_id)


def _assistant_contract(conversation_id: str) -> str:
    return _driver_contract("assistant.txt", conversation_id)


def driver_contract_for(agent_mode: str, conversation_id: str) -> str:
    return _driver_contract(AGENT_MODE_ROLE_PROMPT[agent_mode], conversation_id)


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


def _assistant_prompt(
    user_text: str,
    *,
    is_resume: bool,
    conversation_id: str,
) -> str:
    del is_resume
    return (
        f"{_assistant_contract(conversation_id)}\n\nNewest user message:\n{user_text}\n"
    )


def driver_prompt(
    user_text: str,
    *,
    agent_mode: str,
    conversation_id: str,
) -> str:
    """The first call of a turn, for either agent mode."""
    return (
        f"{driver_contract_for(agent_mode, conversation_id)}"
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
    return driver_repair_prompt(
        unparsed_output,
        agent_mode="orchestrated",
        conversation_id=conversation_id,
    )


def driver_repair_prompt(
    unparsed_output: str,
    *,
    agent_mode: str,
    conversation_id: str,
) -> str:
    """Ask the resumed driving role to repair only its decision envelope."""
    if agent_mode == "single":
        envelope = (
            "Return exactly one JSON object with the fields `action` and "
            "`message`. If the text below is the completed answer, preserve it "
            "in `message` with action `final_answer`. If user input is genuinely "
            "required, use `request_user_input`. There is no `delegate` action "
            "and no `task` field in this mode."
        )
    else:
        envelope = (
            "Return exactly one JSON object with the fields `action`, `message`, "
            "and `task`. If the text below is the completed answer, preserve it "
            "in `message` with action `final_answer`. If user input is genuinely "
            "required, use `request_user_input`; if a separate implementer task "
            "is required, use `delegate`."
        )
    return (
        f"{driver_contract_for(agent_mode, conversation_id)}\n\n"
        "Your previous final output could not be parsed as the required decision "
        f"JSON. Do not redo completed analysis. {envelope} Do not end with "
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
    return driver_continue_prompt(
        action,
        message,
        agent_mode="orchestrated",
        conversation_id=conversation_id,
    )


def driver_continue_prompt(
    action: str,
    message: str,
    *,
    agent_mode: str,
    conversation_id: str,
) -> str:
    """Resume after a non-terminal envelope was emitted as the final item."""
    terminal_actions = (
        "`final_answer` or `request_user_input`"
        if agent_mode == "single"
        else "`final_answer`, `request_user_input`, or `delegate`"
    )
    return (
        f"{driver_contract_for(agent_mode, conversation_id)}\n\n"
        f"Your previous call ended with a non-terminal `{action}` update. The "
        "runtime already showed it to the user. Continue the same work from that "
        "checkpoint without repeating completed analysis. End this call only with "
        f"{terminal_actions}; use `progress` and "
        "`milestone` only for commentary emitted while you keep working.\n\n"
        f"Last update:\n{message}\n"
    )


def implementer_steer_prompt(message: str) -> str:
    """Deliver a user correction to an implementer they interrupted mid-task.

    Unlike `_implementer_prompt` this is not a new delegated task: the session
    being resumed still holds the original one, and the user stopped it
    precisely because they wanted that task done differently. Saying so is what
    keeps the model from re-reading and re-running work it already finished.

    The `user:` marker is load-bearing, not decoration: it is the only thing
    that distinguishes a message the user typed from a task the orchestrator
    delegated, and the implementer's contract keys `reply_user` off exactly
    that. Nothing else in an implementer prompt carries it.
    """
    return (
        f"{_role_prompt('implementer.txt')}\n\n"
        "The user interrupted you while you were working on the task above, and "
        "sent the message below. Read it first: if it asks you something, answer "
        "it with `reply_user`. If it corrects how you were going about the task, "
        "continue from where you stopped — keep the work you already completed, "
        "drop or redo only what the message contradicts, and end with "
        "`final_answer` as usual.\n\n"
        f"user: {message}\n"
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
        message = payload.get("message")
        task = payload.get("task")
        message_is_empty = message is None or (
            isinstance(message, str) and not message.strip()
        )
        task_is_empty = task is None or (isinstance(task, str) and not task.strip())
        if (
            action in {"progress", "milestone"}
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
        if action not in {"final_answer", "reply_user"}:
            continue
        if action == "reply_user" and not allow_reply_user:
            action = "final_answer"
        return {
            "action": action,
            "message": _normalize_orchestrator_text_field(message),
        }
    return {"action": "final_answer", "message": text}


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
    if failure.get("code") in {"codex_call_timeout", "agent_call_timeout"}:
        return (
            f"The {role} produced no output before its idle timeout, so this "
            "turn has no answer. The call reached no decision — this is a "
            "runtime stall, not a model or parsing problem."
        )
    if failure.get("code") == "agent_invalid_output":
        return f"The {role} did not return a valid decision. Continue the conversation to retry."
    if failure.get("code") == "agent_runtime_failure":
        return f"The {role} could not complete its agent call. Check the runner configuration and credentials, then retry."
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
