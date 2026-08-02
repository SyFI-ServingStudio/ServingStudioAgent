"""High-level two-role orchestrator/implementer turn loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ..logging_config import compact_text, log_event
from ..managed_context import write_managed_context
from .codex_cli import run_codex
from .config import (
    DEFAULT_SANDBOX,
    EXECUTION_MODES,
    LOG,
    ORCHESTRATOR_SCHEMA_IN_CONTAINER,
    codex_model,
    normalize_role_runtime,
    workspace_main_for,
)
from .config import container_name as container_name_for
from .docker import container_running, ensure_container
from .prompts import (
    _format_implementer_summaries,
    _implementer_prompt,
    _orchestrator_handoff_prompt,
    _orchestrator_continue_prompt,
    _orchestrator_prompt,
    _orchestrator_repair_prompt,
    compose_final_message,
    parse_orchestrator,
)
from .workspace import prepare_workspace

MAX_ORCHESTRATOR_DECISION_REPAIRS = 2
MAX_ORCHESTRATOR_CONTINUATIONS = 3


async def run_turn(
    workspace_id: str,
    conversation_id: str,
    prompt: str,
    *,
    sandbox: str,
    sessions: dict[str, str] | None = None,
    turn_id: str = "",
    prompt_fingerprint: str = "",
    autonomous: bool = False,
    peer_dir: str | None = None,
    orchestrator_runtime: dict[str, str] | None = None,
    implementer_runtime: dict[str, str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run one user turn through orchestrator/implementer handoffs."""
    turn_id = turn_id or "unknown"
    mode = sandbox if sandbox in EXECUTION_MODES else DEFAULT_SANDBOX
    sessions = sessions or {}
    orchestrator_selection = normalize_role_runtime(
        (orchestrator_runtime or {}).get("model"),
        (orchestrator_runtime or {}).get("effort"),
        (orchestrator_runtime or {}).get("service_tier"),
    )
    implementer_selection = normalize_role_runtime(
        (implementer_runtime or {}).get("model"),
        (implementer_runtime or {}).get("effort"),
        (implementer_runtime or {}).get("service_tier"),
    )
    log_event(
        LOG,
        "turn.run.start",
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        mode=mode,
        prompt_fingerprint=prompt_fingerprint,
        autonomous=autonomous,
        session_roles=sorted(sessions),
    )

    workspace_main_path = workspace_main_for(workspace_id)
    if workspace_main_path.exists():
        yield {"kind": "tool_call", "text": "checking isolated VibeSim workspace..."}
    else:
        yield {"kind": "tool_call", "text": "creating isolated VibeSim workspace..."}
    loop = asyncio.get_event_loop()
    workspace_started = loop.time()
    workspace_task = asyncio.create_task(
        asyncio.to_thread(prepare_workspace, workspace_id),
    )
    while True:
        done, _pending = await asyncio.wait({workspace_task}, timeout=5)
        if done:
            _workspace_main = await workspace_task
            break
        elapsed = loop.time() - workspace_started
        yield {
            "kind": "tool_call",
            "text": f"workspace check still running ({elapsed:.0f}s)...",
        }

    runtime_container_name = container_name_for(workspace_id, conversation_id)
    if await asyncio.to_thread(container_running, runtime_container_name):
        yield {"kind": "tool_call", "text": "checking Docker Codex container..."}
    else:
        yield {"kind": "tool_call", "text": "starting Docker Codex container..."}
    container_started = loop.time()
    container_task = asyncio.create_task(
        asyncio.to_thread(
            ensure_container,
            workspace_id,
            conversation_id,
            _workspace_main,
            mode,
            peer_dir,
            autonomous=autonomous,
            orchestrator_family=codex_model(orchestrator_selection["model"]).family_id,
            implementer_family=codex_model(implementer_selection["model"]).family_id,
        )
    )
    while True:
        done, _pending = await asyncio.wait({container_task}, timeout=10)
        if done:
            container = await container_task
            break
        elapsed = loop.time() - container_started
        yield {
            "kind": "tool_call",
            "text": f"Docker Codex container check still running ({elapsed:.0f}s)...",
        }

    log_event(
        LOG,
        "turn.runtime.ready",
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        workspace=str(_workspace_main),
        container=container,
    )

    orchestrator_session = sessions.get("orchestrator")
    implementer_session = sessions.get("implementer")
    implementer_summaries: list[str] = []
    orchestrator_decision_repairs = 0
    orchestrator_continuations = 0
    # vLLM's constrained text branch can terminate a CodexDS turn at a
    # progress envelope before the model selects its next tool. Keep the
    # existing prompt/parser repair contract, but do not enable constrained
    # decoding for the DeepSeek family.
    orchestrator_output_schema = (
        None
        if codex_model(orchestrator_selection["model"]).family_id == "deepseek"
        else ORCHESTRATOR_SCHEMA_IN_CONTAINER
    )
    next_orchestrator_prompt = _orchestrator_prompt(
        prompt,
        is_resume=bool(orchestrator_session),
        conversation_id=conversation_id,
    )

    while True:
        orchestrator_text: str | None = None
        write_managed_context(
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            role="orchestrator",
        )
        async for ev in run_codex(
            container,
            next_orchestrator_prompt,
            label="orchestrator",
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            model_id=orchestrator_selection["model"],
            effort=orchestrator_selection["effort"],
            service_tier=orchestrator_selection["service_tier"],
            session_id=orchestrator_session,
            output_schema=orchestrator_output_schema,
        ):
            if ev["kind"] == "session":
                orchestrator_session = ev["session_id"]
                yield ev
            elif ev["kind"] == "final":
                orchestrator_text = ev["text"]
            else:
                yield ev
        orchestrator_text = orchestrator_text or ""

        decision = parse_orchestrator(orchestrator_text)
        if decision is None:
            orchestrator_decision_repairs += 1
            log_event(
                LOG,
                "orchestrator.parse_failed",
                conversation_id=conversation_id,
                turn_id=turn_id,
                repair_count=orchestrator_decision_repairs,
                text_preview=compact_text(orchestrator_text),
            )
            if orchestrator_decision_repairs <= MAX_ORCHESTRATOR_DECISION_REPAIRS:
                next_orchestrator_prompt = _orchestrator_repair_prompt(
                    orchestrator_text,
                    conversation_id=conversation_id,
                )
                continue
            sections = []
            if implementer_summaries:
                sections.append(
                    "### Implementer Summary\n\n"
                    f"{_format_implementer_summaries(implementer_summaries)}"
                )
            sections.append("### Error\n\nI could not parse the orchestrator decision.")
            yield {
                "kind": "final",
                "outcome": "final_answer",
                "text": "\n\n".join(sections),
            }
            return

        log_event(
            LOG,
            "orchestrator.decision",
            conversation_id=conversation_id,
            turn_id=turn_id,
            action=decision["action"],
            message_len=len(decision.get("message", "")),
            task_len=len(decision.get("task", "")),
            preview=compact_text(decision.get("message") or decision.get("task") or ""),
        )
        if decision["action"] in {"final_answer", "request_user_input"}:
            yield {
                "kind": "final",
                "outcome": decision["action"],
                "text": compose_final_message(
                    decision["message"],
                    implementer_summaries,
                ),
            }
            return

        if decision["action"] in {"progress", "milestone"}:
            orchestrator_continuations += 1
            yield {
                "kind": "intermediate_output",
                "role": "orchestrator",
                "model": orchestrator_selection["model"],
                "effort": orchestrator_selection["effort"],
                "level": decision["action"],
                "text": decision["message"],
            }
            if orchestrator_continuations > MAX_ORCHESTRATOR_CONTINUATIONS:
                yield {
                    "kind": "final",
                    "outcome": "final_answer",
                    "text": (
                        "The orchestrator repeatedly stopped at a progress "
                        "checkpoint before producing a terminal decision. "
                        "Please continue the conversation to retry."
                    ),
                }
                return
            next_orchestrator_prompt = _orchestrator_continue_prompt(
                decision["action"],
                decision["message"],
                conversation_id=conversation_id,
            )
            continue

        if mode == "read-only":
            body = (
                "This needs an implementer run, but this turn is in read-only mode. "
                "Switch the execution mode to workspace-write if you want me to run it "
                "inside the copied Docker workspace."
            )
            yield {
                "kind": "final",
                "outcome": "final_answer",
                "text": compose_final_message(body, implementer_summaries),
            }
            return

        task = decision["task"]
        # Surface the delegated task itself (the orchestrator→implementer handoff
        # card) rather than a generic "starting..." progress line.
        yield {"kind": "decision", "action": "delegate", "task": task}
        implementer_text: str | None = None
        write_managed_context(
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            role="implementer",
        )
        async for ev in run_codex(
            container,
            _implementer_prompt(
                task,
                is_resume=bool(implementer_session),
            ),
            label="implementer",
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            model_id=implementer_selection["model"],
            effort=implementer_selection["effort"],
            service_tier=implementer_selection["service_tier"],
            session_id=implementer_session,
        ):
            if ev["kind"] == "session":
                implementer_session = ev["session_id"]
                yield ev
            elif ev["kind"] == "final":
                implementer_text = ev["text"]
                yield {"kind": "implementer", "text": implementer_text}
            else:
                yield ev

        if implementer_text is None:
            implementer_text = "(implementer produced no summary)"
        implementer_summaries.append(implementer_text)
        next_orchestrator_prompt = _orchestrator_handoff_prompt(
            task,
            implementer_text,
            conversation_id=conversation_id,
        )
