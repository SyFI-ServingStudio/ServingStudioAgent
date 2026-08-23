"""High-level turn loop for both agent modes.

One `while True` drives every turn. Each round calls the *driving role* — the
orchestrator in `orchestrated` mode, the assistant in `single` mode — parses its
decision envelope, and either terminates the turn, continues after a
non-terminal update, or repairs an unparsable envelope. Only the `delegate` tail
(hand a task to the implementer, then loop back with its summary) is
mode-specific, because `single` has no second role to delegate to.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ..logging_config import compact_text, log_event
from ..managed_context import write_managed_context
from .codex_cli import run_codex
from .config import (
    AGENT_MODE_SCHEMA_IN_CONTAINER,
    DEFAULT_AGENT_MODE,
    DEFAULT_SANDBOX,
    EXECUTION_MODES,
    IMPLEMENTER_SCHEMA_IN_CONTAINER,
    LOG,
    codex_model,
    driving_role_for_agent_mode,
    normalize_agent_mode,
    normalize_role_runtime,
    roles_for_agent_mode,
    workspace_main_for,
)
from .config import container_name as container_name_for
from .docker import container_running, ensure_container
from .prompts import (
    _implementer_prompt,
    _orchestrator_handoff_prompt,
    compose_failure_message,
    compose_final_message,
    driver_continue_prompt,
    driver_prompt,
    driver_repair_prompt,
    implementer_steer_prompt,
    parse_implementer,
    parse_orchestrator,
    transport_failure_reason,
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
    agent_mode: str = DEFAULT_AGENT_MODE,
    peer_dir: str | None = None,
    role_runtimes: dict[str, dict[str, str]] | None = None,
    resume_role: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run one user turn, in either agent mode.

    `resume_role` carries the user's answer to a role they interrupted. Only
    `"implementer"` changes anything: the driving role is already resumed from
    its own session on every turn, so an interrupted orchestrator continues by
    construction. See `_should_resume_implementer` for when the request is
    honored.
    """
    turn_id = turn_id or "unknown"
    mode = sandbox if sandbox in EXECUTION_MODES else DEFAULT_SANDBOX
    agent_mode = normalize_agent_mode(agent_mode)
    single_agent = agent_mode == "single"
    driver_role = driving_role_for_agent_mode(agent_mode)
    sessions = sessions or {}
    role_selections = {
        role: normalize_role_runtime(
            (role_runtimes or {}).get(role, {}).get("model"),
            (role_runtimes or {}).get(role, {}).get("effort"),
            (role_runtimes or {}).get(role, {}).get("service_tier"),
        )
        for role in roles_for_agent_mode(agent_mode)
    }
    driver_selection = role_selections[driver_role]
    log_event(
        LOG,
        "turn.run.start",
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        mode=mode,
        prompt_fingerprint=prompt_fingerprint,
        autonomous=autonomous,
        agent_mode=agent_mode,
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
            agent_mode=agent_mode,
            role_families={
                role: codex_model(selection["model"]).family_id
                for role, selection in role_selections.items()
            },
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

    driver_session = sessions.get(driver_role)
    implementer_session = sessions.get("implementer")
    implementer_summaries: list[str] = []
    orchestrator_decision_repairs = 0
    orchestrator_continuations = 0
    # vLLM's constrained text branch can terminate a CodexDS turn at a
    # progress envelope before the model selects its next tool. Keep the
    # existing prompt/parser repair contract, but do not enable constrained
    # decoding for the DeepSeek family.
    driver_output_schema = (
        None
        if codex_model(driver_selection["model"]).family_id == "deepseek"
        else AGENT_MODE_SCHEMA_IN_CONTAINER[agent_mode]
    )
    next_driver_prompt = driver_prompt(
        prompt,
        agent_mode=agent_mode,
        conversation_id=conversation_id,
    )
    # A task waiting for the implementer, and the prompt that delivers it. Set
    # here only when the user is answering an implementer they interrupted;
    # otherwise the loop fills it from a `delegate` decision. Either way the
    # implementer's summary comes back through the same handoff, so an
    # interrupted turn resumes rather than restarting.
    pending_task: str | None = None
    pending_implementer_prompt: str | None = None
    # Whether the implementer may end this turn by answering the user itself.
    # True only where the prompt carries the user's own words — a delegated task
    # comes from the orchestrator, so there is nobody there to reply to.
    pending_implementer_may_reply = False
    if _should_resume_implementer(
        resume_role,
        agent_mode=agent_mode,
        mode=mode,
        sessions=sessions,
    ):
        log_event(
            LOG,
            "turn.resume_implementer",
            conversation_id=conversation_id,
            turn_id=turn_id,
            codex_session_id=implementer_session,
        )
        pending_task = prompt
        pending_implementer_prompt = implementer_steer_prompt(prompt)
        pending_implementer_may_reply = True

    while True:
        if pending_task is not None:
            assert pending_implementer_prompt is not None
            implementer_outcome: dict[str, Any] = {
                "text": None,
                "reply": None,
                "failure": None,
                "session": implementer_session,
            }
            async for ev in _run_implementer(
                container,
                pending_implementer_prompt,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                selection=role_selections["implementer"],
                session_id=implementer_session,
                outcome=implementer_outcome,
                allow_reply_user=pending_implementer_may_reply,
            ):
                yield ev
            implementer_session = implementer_outcome["session"]
            implementer_failure = implementer_outcome["failure"]

            # Handing a placeholder summary back to the orchestrator would spend
            # one more call — against the same unavailable gateway — to conclude
            # nothing.
            if implementer_failure is not None:
                log_event(
                    LOG,
                    "turn.transport_failed",
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    role="implementer",
                    code=implementer_failure["code"],
                    status=implementer_failure["status"],
                    implementer_rounds=len(implementer_summaries),
                )
                yield {
                    "kind": "final",
                    "failure": implementer_failure,
                    "text": compose_failure_message(
                        transport_failure_reason("implementer", implementer_failure),
                        implementer_summaries,
                    ),
                }
                return

            # The user asked the implementer something and it answered, so the
            # turn is over: routing this through the orchestrator would spend a
            # call to have a third party repeat it. The cost is that the
            # orchestrator's session never sees this exchange — acceptable
            # because it is an aside about work the orchestrator already
            # delegated, and the next delegation restates the task anyway.
            implementer_reply = implementer_outcome["reply"]
            if implementer_reply:
                log_event(
                    LOG,
                    "turn.implementer_replied",
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    reply_len=len(implementer_reply),
                    reply_preview=compact_text(implementer_reply),
                )
                yield {
                    "kind": "final",
                    "text": implementer_reply,
                    "outcome": "final_answer",
                    # Who answered, so the next message can continue with them.
                    # Only a direct reply carries this: every other ending goes
                    # through the driving role, which is the default anyway.
                    "role": "implementer",
                }
                return

            implementer_text = (
                implementer_outcome["text"] or "(implementer produced no summary)"
            )
            implementer_summaries.append(implementer_text)
            next_driver_prompt = _orchestrator_handoff_prompt(
                pending_task,
                implementer_text,
                conversation_id=conversation_id,
            )
            pending_task = None
            pending_implementer_prompt = None
            pending_implementer_may_reply = False
            continue

        orchestrator_text: str | None = None
        orchestrator_failure: dict[str, Any] | None = None
        # Held rather than forwarded: `usage` is what closes this call's card in
        # the UI, and a checkpoint decision produces one more message *after* the
        # call has ended. Releasing usage on arrival would push that closing
        # message into the next round's card, splitting one call across two.
        driver_usage: dict[str, Any] | None = None
        write_managed_context(
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            role=driver_role,
        )
        async for ev in run_codex(
            container,
            next_driver_prompt,
            label=driver_role,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            model_id=driver_selection["model"],
            effort=driver_selection["effort"],
            service_tier=driver_selection["service_tier"],
            session_id=driver_session,
            output_schema=driver_output_schema,
        ):
            if ev["kind"] == "session":
                driver_session = ev["session_id"]
                yield ev
            elif ev["kind"] == "final":
                orchestrator_text = ev["text"]
                orchestrator_failure = ev.get("failure")
            elif ev["kind"] == "usage":
                driver_usage = ev
            else:
                yield ev
        orchestrator_text = orchestrator_text or ""

        # A call that never reached the model has no decision to repair. Retrying
        # would spend another full reconnect cycle (or idle timeout) per round on
        # the same outage, and would still end here.
        if orchestrator_failure is not None:
            log_event(
                LOG,
                "turn.transport_failed",
                conversation_id=conversation_id,
                turn_id=turn_id,
                role=driver_role,
                code=orchestrator_failure["code"],
                status=orchestrator_failure["status"],
                implementer_rounds=len(implementer_summaries),
            )
            if driver_usage is not None:
                yield driver_usage
            yield {
                "kind": "final",
                "failure": orchestrator_failure,
                "text": compose_failure_message(
                    transport_failure_reason(driver_role, orchestrator_failure),
                    implementer_summaries,
                ),
            }
            return

        decision = parse_orchestrator(
            orchestrator_text,
            allow_delegate=not single_agent,
        )
        if decision is None:
            orchestrator_decision_repairs += 1
            log_event(
                LOG,
                "orchestrator.parse_failed",
                conversation_id=conversation_id,
                turn_id=turn_id,
                role=driver_role,
                repair_count=orchestrator_decision_repairs,
                text_preview=compact_text(orchestrator_text),
            )
            # An unparseable answer has no closing message to place, so this
            # call's card closes here on either branch.
            if driver_usage is not None:
                yield driver_usage
            if orchestrator_decision_repairs <= MAX_ORCHESTRATOR_DECISION_REPAIRS:
                next_driver_prompt = driver_repair_prompt(
                    orchestrator_text,
                    agent_mode=agent_mode,
                    conversation_id=conversation_id,
                )
                continue
            yield {
                "kind": "final",
                "outcome": "final_answer",
                "text": compose_failure_message(
                    f"I could not parse the {driver_role} decision after "
                    f"{MAX_ORCHESTRATOR_DECISION_REPAIRS} repair attempts.",
                    implementer_summaries,
                ),
            }
            return

        log_event(
            LOG,
            "orchestrator.decision",
            conversation_id=conversation_id,
            turn_id=turn_id,
            role=driver_role,
            action=decision["action"],
            message_len=len(decision.get("message", "")),
            task_len=len(decision.get("task", "")),
            preview=compact_text(decision.get("message") or decision.get("task") or ""),
        )
        # A checkpoint's message is this call's last word, so it goes out before
        # the usage event that closes the call — see `driver_usage` above.
        checkpoint = decision["action"] in {"progress", "milestone"}
        if checkpoint:
            yield {
                "kind": "intermediate_output",
                "role": driver_role,
                "model": driver_selection["model"],
                "effort": driver_selection["effort"],
                "level": decision["action"],
                "text": decision["message"],
            }
        if driver_usage is not None:
            yield driver_usage

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

        if checkpoint:
            orchestrator_continuations += 1
            if orchestrator_continuations > MAX_ORCHESTRATOR_CONTINUATIONS:
                yield {
                    "kind": "final",
                    "outcome": "final_answer",
                    "text": (
                        f"The {driver_role} repeatedly stopped at a progress "
                        "checkpoint before producing a terminal decision. "
                        "Please continue the conversation to retry."
                    ),
                }
                return
            next_driver_prompt = driver_continue_prompt(
                decision["action"],
                decision["message"],
                agent_mode=agent_mode,
                conversation_id=conversation_id,
            )
            continue

        # Only `delegate` can reach here, and only in orchestrated mode: the
        # single-agent parser refuses that action, so its envelope is either
        # terminal or a continuation and the loop never falls through.
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
        pending_task = task
        pending_implementer_prompt = _implementer_prompt(
            task,
            is_resume=bool(implementer_session),
        )


async def _run_implementer(
    container: str,
    prompt: str,
    *,
    workspace_id: str,
    conversation_id: str,
    turn_id: str,
    selection: dict[str, str],
    session_id: str | None,
    outcome: dict[str, Any],
    allow_reply_user: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Stream one implementer call, reporting its result through `outcome`.

    An async generator cannot return a value alongside its stream, so the caller
    passes `outcome` and reads `text` / `reply` / `failure` / `session` back out
    of it. Both entries into the implementer — the loop's `delegate` tail and a
    user answering an implementer they interrupted — go through here, so the
    session threading and the failure shape cannot drift apart.

    `outcome["reply"]` is set only when the implementer answered the user
    directly, which is also the only case where it is not a summary for the
    orchestrator: it ends the turn, so it is not appended to the handoff and
    does not become an `implementer` card.
    """
    write_managed_context(
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        role="implementer",
    )
    async for ev in run_codex(
        container,
        prompt,
        label="implementer",
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        model_id=selection["model"],
        effort=selection["effort"],
        service_tier=selection["service_tier"],
        session_id=session_id,
        output_schema=_implementer_output_schema(selection),
    ):
        if ev["kind"] == "session":
            outcome["session"] = ev["session_id"]
            yield ev
        elif ev["kind"] == "final":
            failure = ev.get("failure")
            outcome["failure"] = failure
            if failure is None:
                result = parse_implementer(
                    ev["text"], allow_reply_user=allow_reply_user
                )
                if result["action"] == "reply_user":
                    outcome["reply"] = result["message"]
                else:
                    outcome["text"] = result["message"]
                    yield {"kind": "implementer", "text": result["message"]}
        else:
            yield ev


def _implementer_output_schema(selection: dict[str, str]) -> str | None:
    """Constrain the implementer envelope, except where that breaks the model.

    Same carve-out as the driver: vLLM's constrained text branch can end a
    DeepSeek turn early, so that family keeps the prompt contract alone and
    leans on `parse_implementer`'s fallback.
    """
    if codex_model(selection["model"]).family_id == "deepseek":
        return None
    return IMPLEMENTER_SCHEMA_IN_CONTAINER


def _should_resume_implementer(
    resume_role: str | None,
    *,
    agent_mode: str,
    mode: str,
    sessions: dict[str, str],
) -> bool:
    """Whether this turn should open at the implementer instead of the driver.

    Every condition is a way the request can be stale rather than a policy
    choice, so failing any of them falls back to a normal turn rather than
    erroring — the driving role still has the conversation and can re-delegate:

    - only the implementer needs rerouting; the driving role resumes its own
      session on every turn already;
    - `single` mode has no implementer at all;
    - the session can disappear between the interrupt and the reply if the user
      switches the implementer to another model family, which makes the rollout
      unresumable (`store.sessions_for_prompt` drops it);
    - read-only turns refuse implementer work outright.
    """
    if resume_role != "implementer":
        return False
    if agent_mode == "single" or mode == "read-only":
        return False
    return bool(sessions.get("implementer"))
