"""Single and orchestrated role policies over the same provider call lifecycle."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.decisions import parse_implementer, parse_orchestrator
from ..domain.errors import ProviderUnavailable
from ..domain.evidence import AnalyzerTurnContext, prompt_with_analyzer_context
from ..domain.roles import AgentMode, Role, Sandbox
from ..domain.turns import Outcome, TurnInput, TurnResult
from ..prompts.render import Prompts
from ..providers.base import AgentRequest, Selection
from ..providers.registry import ProviderRegistry
from ..runtime.execution import Execution


@dataclass(frozen=True)
class _CallResult:
    final: dict[str, Any] | None
    usage: dict[str, Any] | None


class ConversationDriver:
    def __init__(
        self,
        providers: ProviderRegistry,
        prompts: Prompts,
        *,
        prepare: Callable[[TurnInput], Awaitable[Execution]],
        before_call: Callable[[AgentRequest], Awaitable[None]],
        after_turn: Callable[[TurnInput], None] | None = None,
    ):
        self.providers = providers
        self.prompts = prompts
        self.prepare = prepare
        self.before_call = before_call
        self.after_turn = after_turn

    def selection(self, request: TurnInput, role: Role) -> Selection:
        runtime = request.runtimes[role]
        selection = self.providers.select(
            runtime.provider_id,
            runtime.model_id,
            effort=runtime.effort,
            service_tier=runtime.service_tier,
        )
        if selection.session_scope != runtime.session_scope:
            raise ValueError("stored role scope is incompatible with provider")
        if not self.providers.available(selection.provider_id):
            raise ProviderUnavailable((selection.provider_id,))
        if request.sessions.get(role) and not selection.model.resumable:
            raise ValueError("selected model cannot resume this conversation")
        return selection

    def validate(self, request: TurnInput) -> None:
        if not set(request.mode.roles).issubset(request.runtimes):
            raise ValueError("all active role selections are required")
        for role in request.mode.roles:
            self.selection(request, role)

    async def _call(
        self,
        request: TurnInput,
        *,
        role: Role,
        prompt: str,
        execution: Execution,
        selection: Selection,
        sessions: dict[Role, str],
        hold_usage: bool,
    ) -> AsyncIterator[dict[str, Any] | _CallResult]:
        call = AgentRequest(
            request.workspace_id,
            request.conversation_id,
            request.turn_id,
            role,
            prompt,
            execution,
            selection,
            session_id=sessions.get(role),
            # From the execution: the same file is a mount target in a
            # container and a state-root path on the host, and only the
            # execution knows which side this turn is running on.
            output_schema=Path(execution.schema_directory) / f"{role.value}.schema.json",
        )
        # Invalidate the previous role's ready state before preparation can yield.
        yield {"kind": "role_start", "role": role.value}
        await self.before_call(call)
        final = usage = None
        async with aclosing(self.providers.run(call)) as events:
            async for event in events:
                kind = event["kind"]
                if (
                    kind in {"role_start", "role_ready", "session"}
                    and event.get("role") != role.value
                ):
                    raise ValueError("provider event belongs to a different role")
                if kind == "role_start":
                    continue
                if kind == "session":
                    sessions[role] = event["session_id"]
                    yield event
                elif kind == "final":
                    final = event
                elif kind == "usage" and hold_usage:
                    usage = event
                else:
                    yield event
        yield _CallResult(final, usage)

    @staticmethod
    def _result(
        outcome: Outcome,
        text: str,
        summaries: list[str],
        *,
        failure: dict | None = None,
        resume_role: Role | None = None,
    ) -> TurnResult:
        metadata = {"implementer_summaries": list(summaries)} if summaries else {}
        if failure is not None:
            metadata["failure"] = failure
        if outcome is Outcome.FAILED and summaries:
            count = len(summaries)
            noun = "round" if count == 1 else "rounds"
            text = (
                f"{text.strip()}\n\n{count} implementer {noun} completed before this failure and are shown "
                "as their own cards above. The agent sessions are preserved; continue "
                "the conversation to resume from this point."
            )
        return TurnResult(outcome, text, metadata, resume_role=resume_role)

    async def run(
        self, request: TurnInput
    ) -> AsyncIterator[dict[str, Any] | TurnResult]:
        try:
            async with aclosing(self._run(request)) as events:
                async for event in events:
                    yield event
        finally:
            if self.after_turn is not None:
                self.after_turn(request)

    async def _run(
        self, request: TurnInput
    ) -> AsyncIterator[dict[str, Any] | TurnResult]:
        self.validate(request)
        selections = {
            role: self.selection(request, role) for role in request.mode.roles
        }
        execution = await self.prepare(request)
        context = (
            AnalyzerTurnContext.model_validate(request.analyzer_context)
            if request.analyzer_context is not None
            else None
        )
        user_text = prompt_with_analyzer_context(request.text, context)
        prompt = self.prompts.driver_prompt(
            user_text, mode=request.mode, conversation_id=request.conversation_id
        )
        sessions = dict(request.sessions)
        driving_role = request.mode.driver
        summaries: list[str] = []
        repairs = continuations = 0
        pending_task = None
        may_reply = False
        if (
            request.mode is AgentMode.ORCHESTRATED
            and request.resume_role == "implementer"
            and request.sandbox is not Sandbox.READ_ONLY
            and sessions.get(Role.IMPLEMENTER)
        ):
            pending_task = user_text
            prompt = self.prompts.implementer_steer_prompt(user_text)
            may_reply = True
        while True:
            role = Role.IMPLEMENTER if pending_task is not None else driving_role
            result = None
            async with aclosing(
                self._call(
                    request,
                    role=role,
                    prompt=prompt,
                    execution=execution,
                    selection=selections[role],
                    sessions=sessions,
                    hold_usage=role is driving_role,
                )
            ) as events:
                async for event in events:
                    if isinstance(event, _CallResult):
                        result = event
                    else:
                        yield event
            assert result is not None
            text = (result.final or {}).get("text", "")
            failure = (result.final or {}).get("failure")
            if result.final is None or failure:
                if result.usage is not None:
                    yield result.usage
                yield self._result(
                    Outcome.FAILED,
                    text or "The agent returned no result.",
                    summaries,
                    failure=failure or {"code": "agent_runtime_failure"},
                )
                return
            if role is Role.IMPLEMENTER:
                decision = parse_implementer(text, allow_reply_user=may_reply)
                if decision["action"] == "reply_user":
                    yield self._result(
                        Outcome.ANSWER, decision["message"], summaries, resume_role=role
                    )
                    return
                summaries.append(decision["message"])
                yield {"kind": "implementer", "text": decision["message"]}
                prompt = self.prompts.orchestrator_handoff_prompt(
                    pending_task,
                    decision["message"],
                    conversation_id=request.conversation_id,
                )
                pending_task = None
                may_reply = False
                continue
            decision = parse_orchestrator(
                text, allow_delegate=request.mode is AgentMode.ORCHESTRATED
            )
            if decision is None:
                if result.usage is not None:
                    yield result.usage
                repairs += 1
                if repairs > 2:
                    yield self._result(
                        Outcome.FAILED,
                        f"The {role.value} did not return a valid decision after two repair attempts.",
                        summaries,
                    )
                    return
                prompt = self.prompts.driver_repair_prompt(
                    text,
                    agent_mode=request.mode.value,
                    conversation_id=request.conversation_id,
                )
                continue
            action = decision["action"]
            if action in {"progress", "milestone"}:
                yield {
                    "kind": "intermediate_output",
                    "role": role.value,
                    "model": selections[role].model.model_id,
                    "effort": selections[role].effort,
                    "level": action,
                    "text": decision["message"],
                }
            if result.usage is not None:
                yield result.usage
            if action in {"final_answer", "request_user_input"}:
                yield self._result(
                    Outcome(action), decision["message"].strip(), summaries
                )
                return
            if action in {"progress", "milestone"}:
                continuations += 1
                if continuations > 3:
                    yield self._result(
                        Outcome.FAILED,
                        f"The {role.value} repeatedly stopped at a progress checkpoint. Continue the conversation to retry.",
                        summaries,
                    )
                    return
                prompt = self.prompts.driver_continue_prompt(
                    action,
                    decision["message"],
                    agent_mode=request.mode.value,
                    conversation_id=request.conversation_id,
                )
                continue
            if request.sandbox is Sandbox.READ_ONLY:
                yield self._result(
                    Outcome.ANSWER,
                    "This needs an implementer run, but this turn is in read-only mode. "
                    "Switch the execution mode to workspace-write if you want me to run it "
                    "inside the copied Docker workspace.",
                    summaries,
                )
                return
            pending_task = decision["task"]
            yield {"kind": "decision", "action": "delegate", "task": pending_task}
            prompt = self.prompts.implementer_prompt(
                pending_task, is_resume=bool(sessions.get(Role.IMPLEMENTER))
            )
