"""Durable background turns whose lifetime does not depend on a subscriber."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from typing import Any, Protocol
from uuid import uuid4

from ..domain.events import MANAGED_JOB_EVENTS
from ..domain.evidence import (
    AnalyzerTurnContext,
    freeze_citations,
    latest_citation_dictionary,
    persisted_context,
)
from ..domain.roles import AgentMode, Role, Sandbox
from ..domain.turns import Outcome, TurnInput, TurnOptions, TurnResult
from ..storage.conversations import Conversations
from ..storage.sessions import Session, Sessions
from ..storage.turns import Turns
from .naming import NamingService

_TURN_FAILURES = {
    "agent_call_timeout": "The Agent stalled without producing output. Continue the conversation to retry.",
    "agent_invalid_output": "The Agent did not return a valid decision. Continue the conversation to retry.",
    "runtime_storage_full": (
        "The Agent runtime could not start because the host disk is full. "
        "Free space, then retry this question."
    ),
    "upstream_unavailable": (
        "The upstream model service is unavailable, so the Agent could not "
        "produce an answer. This is a service outage, not a problem with your "
        "question. The conversation is intact \u2014 continue it to retry."
    ),
    "upstream_rate_limited": (
        "The upstream model service is rate limiting this account, so the Agent "
        "could not produce an answer. The conversation is intact \u2014 wait a "
        "moment, then continue it to retry."
    ),
    "codex_call_timeout": (
        "The Agent stalled without producing output and the call was stopped. "
        "The conversation is intact \u2014 continue it to retry."
    ),
    "agent_runtime_failure": (
        "The Agent runtime failed before producing an answer. "
        "Retry the question; the full diagnostic is available in the backend log."
    ),
}


class TurnDriver(Protocol):
    def validate(self, request: TurnInput) -> None: ...

    def run(self, request: TurnInput) -> AsyncIterator[dict[str, Any] | TurnResult]: ...


@dataclass(frozen=True)
class TurnStorage:
    conversations: Conversations
    sessions: Sessions
    turns: Turns


def history_activity(events: list[dict[str, Any]], workspace_id: str) -> tuple[list[dict], list[dict]]:
    """Project durable event order into the existing browser history shape."""
    activity = []
    intermediate = []
    for event in events:
        kind, payload = event["kind"], event["payload"]
        if kind == "intermediate_output":
            output = {
                "role": str(payload.get("role") or "orchestrator"),
                "model": str(payload.get("model") or ""),
                "effort": str(payload.get("effort") or ""),
                "level": "milestone" if payload.get("level") == "milestone" else "progress",
                "text": str(payload.get("text") or ""),
            }
            intermediate.append(output)
            activity.append({"kind": kind, **output})
        elif kind == "decision":
            activity.append({"kind": kind, "action": str(payload.get("action") or ""),
                             "task": str(payload.get("task") or "")})
        elif kind == "implementer":
            activity.append({"kind": kind, "text": str(payload.get("text") or "")})
        elif kind == "usage":
            activity.append({"kind": kind, "role": str(payload.get("role") or ""),
                             "model": str(payload.get("model") or ""),
                             "effort": str(payload.get("effort") or ""),
                             "duration_ms": int(payload.get("duration_ms") or 0),
                             "tokens": payload.get("tokens") or {}})
        elif kind in MANAGED_JOB_EVENTS:
            job = {"kind": "job", "workspaceId": str(payload.get("workspaceId") or workspace_id),
                   "status": str(payload.get("status") or kind),
                   "experimentId": str(payload.get("experimentId") or ""),
                   "experimentPath": str(payload.get("experimentPath") or ""),
                   "jobId": str(payload.get("jobId") or "")}
            if payload.get("jobKind"):
                job.update({"jobKind": str(payload["jobKind"]),
                            "resourceId": str(payload.get("resourceId") or ""),
                            "analyzerResourceId": str(payload.get("analyzerResourceId") or ""),
                            "artifactPath": str(payload.get("artifactPath") or ""),
                            "descriptor": payload.get("descriptor") or {},
                            "summary": payload.get("summary")})
            activity.append(job)
    return activity, intermediate


@dataclass
class TurnHandle:
    request: TurnInput
    task: asyncio.Task[TurnResult] | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    finished: bool = False
    started: bool = False
    cancel_requested: bool = False
    cancel_sent: bool = False
    cancel_resume_role: str = ""
    ready: bool = True
    role: str = ""
    error: BaseException | None = None
    cancel_timer: asyncio.Task | None = None

    def notify(self) -> None:
        previous = self.changed
        self.changed = asyncio.Event()
        previous.set()


class TurnService:
    def __init__(self, storage: Callable[[str], TurnStorage], driver: TurnDriver, *,
                 safe_interrupt_timeout: float = 30, logger: logging.Logger,
                 fingerprint: Callable[[TurnInput], str] | None = None,
                 naming: NamingService | None = None):
        self.storage = storage
        self.driver = driver
        self.safe_interrupt_timeout = safe_interrupt_timeout
        self.logger = logger
        self.fingerprint = fingerprint
        self.naming = naming
        self._active: dict[tuple[str, str], TurnHandle] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._closing = False
        self._deleting: set[tuple[str, str]] = set()
        self._deletions: dict[tuple[str, str], asyncio.Task] = {}

    def current(self, workspace_id: str, conversation_id: str) -> TurnHandle | None:
        return self._active.get((workspace_id, conversation_id))

    def start(self, workspace_id: str, conversation_id: str, text: str,
              *, options: TurnOptions | None = None,
              after_execution: Callable[[TurnInput], Awaitable[None]] | None = None,
              auto_name: bool = True) -> TurnHandle:
        if self._closing:
            raise RuntimeError("turn service is shutting down")
        if not text.strip():
            raise ValueError("message must not be empty")
        text = text.strip()
        options = options or TurnOptions()
        key = (workspace_id, conversation_id)
        if key in self._deleting:
            raise ValueError("conversation is being deleted")
        if key in self._active:
            raise ValueError("conversation already has an active turn")
        store = self.storage(workspace_id)
        conversation = store.conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        if store.turns.active(conversation_id) is not None:
            raise ValueError("conversation has an unfinished turn requiring recovery")
        has_history = bool(store.conversations.messages(conversation_id, limit=1))
        mode = AgentMode(conversation["agent_mode"]) if has_history or options.mode is None else options.mode
        autonomous = bool(conversation["autonomous"]) if has_history or options.autonomous is None else options.autonomous
        resume_role = conversation["interrupted_role"] if options.resume_role is None else str(options.resume_role)
        if resume_role and Role(resume_role) not in mode.roles:
            raise ValueError("resume role is inactive in this mode")
        context = (AnalyzerTurnContext.model_validate(options.analyzer_context)
                   if options.analyzer_context is not None else None)
        runtimes = store.conversations.runtimes(conversation_id)
        request = TurnInput(workspace_id, conversation_id, uuid4().hex, text, mode,
                            runtimes, store.sessions.compatible(conversation_id,
                                {role: runtime.session_scope for role, runtime in runtimes.items()}),
                            resume_role,
                            sandbox=options.sandbox or Sandbox(conversation["sandbox"]),
                            autonomous=autonomous,
                            peer_workspace=conversation["peer_workspace"],
                            prompt_fingerprint=conversation["prompt_fingerprint"],
                            analyzer_context=persisted_context(context))
        self.driver.validate(request)
        if self.fingerprint is not None:
            request = replace(request, prompt_fingerprint=self.fingerprint(request))
        store.turns.start(conversation_id, request.turn_id, text,
                          metadata={"analyzer_context": request.analyzer_context} if context is not None else None,
                          settings={"sandbox": request.sandbox.value, "autonomous": int(request.autonomous),
                                    "agent_mode": request.mode.value, "interrupted_role": request.resume_role,
                                    "prompt_fingerprint": request.prompt_fingerprint})
        handle = TurnHandle(request)
        self._active[key] = handle
        handle.task = asyncio.create_task(self._run(handle, store, after_execution, auto_name), name=f"turn-{request.turn_id}")
        # Retrieve errors even when all clients disconnected; wait/stream still raise.
        handle.task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        return handle

    async def wait(self, handle: TurnHandle) -> TurnResult:
        assert handle.task is not None
        return await asyncio.shield(handle.task)

    def cancel(self, workspace_id: str, conversation_id: str, turn_id: str) -> bool:
        handle = self._active.get((workspace_id, conversation_id))
        if handle is None or handle.request.turn_id != turn_id:
            return False
        if not handle.cancel_requested:
            handle.cancel_requested = True
            self._cancel_if_ready(handle)
            if not handle.cancel_sent:
                handle.cancel_timer = asyncio.create_task(self._force_cancel(handle))
        return True

    def _cancel_if_ready(self, handle: TurnHandle) -> None:
        if handle.started and handle.ready and handle.cancel_requested and not handle.cancel_sent:
            self._send_cancel(handle)

    @staticmethod
    def _send_cancel(handle: TurnHandle) -> None:
        handle.cancel_resume_role = handle.role if handle.ready else ""
        handle.cancel_sent = True
        assert handle.task is not None
        handle.task.cancel()

    async def _force_cancel(self, handle: TurnHandle) -> None:
        await asyncio.sleep(self.safe_interrupt_timeout)
        if not handle.finished and not handle.cancel_sent and handle.started:
            self._send_cancel(handle)

    async def stream(self, workspace_id: str, conversation_id: str, turn_id: str,
                     *, after_sequence: int = -1) -> AsyncIterator[dict[str, Any]]:
        store = self.storage(workspace_id)
        if store.turns.get(conversation_id, turn_id) is None:
            raise KeyError(turn_id)
        handle = self._active.get((workspace_id, conversation_id))
        if handle is not None and handle.request.turn_id != turn_id:
            handle = None
        while True:
            changed = handle.changed if handle is not None else None
            finished_before_read = handle is None or handle.finished
            for event in store.turns.events(conversation_id, turn_id, after_sequence=after_sequence):
                after_sequence = event["sequence"]
                yield event
            # Yielding gives the producer time to commit more events. Only stop
            # after a read that began with a stable, already-finished producer.
            if finished_before_read:
                if handle is not None and handle.error is not None:
                    raise RuntimeError("turn finalization failed") from handle.error
                if handle is None:
                    durable = store.turns.get(conversation_id, turn_id)
                    if durable is None:
                        raise KeyError(turn_id)
                    if durable["status"] == "running":
                        raise RuntimeError("turn requires recovery before streaming can complete")
                return
            assert changed is not None
            await changed.wait()

    async def _run(self, handle: TurnHandle, store: TurnStorage,
                   after_execution: Callable[[TurnInput], Awaitable[None]] | None = None,
                   auto_name: bool = True) -> TurnResult:
        request = handle.request
        handle.started = True
        result = TurnResult(Outcome.FAILED, "The agent did not produce a result.")
        execution_failed = False
        naming_ticket = None
        try:
            try:
                if handle.cancel_requested:
                    raise asyncio.CancelledError
                async with self._locks.setdefault(request.workspace_id, asyncio.Lock()):
                    pending_failure = None
                    try:
                        async with aclosing(self.driver.run(request)) as events:
                            async for event in events:
                                if isinstance(event, TurnResult):
                                    result = event
                                    execution_failed = result.outcome is Outcome.FAILED
                                    break
                                kind = event["kind"]
                                if kind == "done":
                                    raise ValueError("turn terminal events belong to the service")
                                if kind == "role_start":
                                    handle.role = Role(event["role"]).value
                                    handle.ready = False
                                elif kind == "role_ready":
                                    if event["role"] != handle.role:
                                        raise ValueError("readiness role differs from active role")
                                    handle.ready = True
                                elif kind == "session":
                                    role = Role(event["role"])
                                    runtime = request.runtimes[role]
                                    store.sessions.save(request.conversation_id, Session(
                                        role, runtime.provider_id, runtime.session_scope, event["session_id"]))
                                store.turns.append_event(request.turn_id, kind, event)
                                handle.notify()
                                self._cancel_if_ready(handle)
                                if handle.cancel_sent:
                                    await asyncio.sleep(0)
                    except Exception as error:
                        pending_failure = error
                        raise
                    finally:
                        if after_execution is not None:
                            try:
                                await self._finish_execution(after_execution, request)
                            except asyncio.CancelledError:
                                if pending_failure is not None:
                                    raise pending_failure
                                raise
            except asyncio.CancelledError:
                if not execution_failed:
                    result = TurnResult(Outcome.CANCELLED, "Stopped.")
            except Exception as error:
                execution_failed = True
                self.logger.exception("Turn execution failed", extra={"turn_id": request.turn_id})
                code = "runtime_storage_full" if "no space left on device" in str(error).lower() else "agent_runtime_failure"
                result = TurnResult(Outcome.FAILED, "The agent could not complete this turn.",
                                    metadata={"failure": {"code": code}})
            if handle.cancel_requested and not execution_failed:
                result = TurnResult(Outcome.CANCELLED, "Stopped.")
            if result.outcome is Outcome.CANCELLED:
                interrupted = handle.cancel_resume_role
            elif result.outcome is Outcome.FAILED:
                conversation = store.conversations.get(request.conversation_id)
                if conversation is None:
                    raise KeyError(request.conversation_id)
                interrupted = conversation["interrupted_role"]
            else:
                interrupted = result.resume_role.value if result.resume_role is not None else ""
            context = (AnalyzerTurnContext.model_validate(request.analyzer_context)
                       if request.analyzer_context is not None else None)
            events = store.turns.events(request.conversation_id, request.turn_id)
            dictionary = latest_citation_dictionary(events, context.citation_dictionary if context is not None else None)
            activity, intermediate = history_activity(events, request.workspace_id)
            activity.append({"kind": "error" if result.outcome is Outcome.FAILED else "final",
                             "text": result.text,
                             **({"outcome": result.outcome.value} if result.outcome is not Outcome.FAILED else {})})
            failure = None
            if result.outcome is Outcome.FAILED:
                raw_failure = result.metadata.get("failure")
                code = raw_failure.get("code") if isinstance(raw_failure, dict) else None
                code = code if isinstance(code, str) and code in _TURN_FAILURES else "agent_runtime_failure"
                failure = {"code": code, "message": _TURN_FAILURES[code]}
            metadata = {"citations": freeze_citations(result.text, dictionary),
                        "citation_dictionary_id": dictionary.identity if dictionary is not None else None,
                        "citation_dsl_version": "v2" if dictionary is not None else None,
                        **result.metadata, "activity": activity,
                        "intermediate_outputs": intermediate, "failure": failure,
                        "outcome": result.outcome.value,
                        "interrupted_role": interrupted}
            if auto_name and self.naming is not None:
                naming_ticket = self.naming.prepare(request, result)
            metadata["naming_scheduled"] = naming_ticket is not None
            done = {**metadata, "kind": "done", "turn_id": request.turn_id,
                    "text": result.text}
            result = TurnResult(result.outcome, result.text, metadata=metadata,
                                resume_role=Role(interrupted) if interrupted else None)
            store.turns.finish(request.conversation_id, request.turn_id, text=result.text,
                               status="failed" if result.outcome is Outcome.FAILED else "complete",
                               interrupted_role=interrupted, metadata=metadata, terminal_event=done)
            if naming_ticket is not None:
                naming_ticket.commit()
            return result
        except BaseException as error:
            if naming_ticket is not None:
                naming_ticket.abort()
            handle.error = error
            raise
        finally:
            if handle.cancel_timer is not None:
                handle.cancel_timer.cancel()
            handle.finished = True
            self._active.pop((request.workspace_id, request.conversation_id), None)
            handle.notify()

    @staticmethod
    async def _finish_execution(callback, request) -> None:
        task = asyncio.create_task(callback(request))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        # A cleanup error must win over a concurrent Stop and remain FAILED.
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def delete(self, workspace_id: str, conversation_id: str,
                     cleanup: Callable[[str, str], Awaitable[None]]) -> None:
        if self._closing:
            raise RuntimeError("turn service is shutting down")
        key = (workspace_id, conversation_id)
        existing = self._deletions.get(key)
        if existing is not None:
            await asyncio.shield(existing)
            return
        store = self.storage(workspace_id)
        if store.conversations.get(conversation_id) is None:
            return
        self._deleting.add(key)

        async def remove():
            try:
                handle = self.current(workspace_id, conversation_id)
                if handle is not None:
                    self.cancel(workspace_id, conversation_id, handle.request.turn_id)
                    await self.wait(handle)
                async with self._locks.setdefault(workspace_id, asyncio.Lock()):
                    await cleanup(workspace_id, conversation_id)
                    store.conversations.delete(conversation_id)
            finally:
                self._deleting.discard(key)
                self._deletions.pop(key, None)

        removal = remove()
        try:
            task = asyncio.create_task(removal)
        except BaseException:
            removal.close()
            self._deleting.discard(key)
            raise
        self._deletions[key] = task
        task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        await asyncio.shield(task)

    async def close(self) -> None:
        self._closing = True
        handles = list(self._active.values())
        for handle in handles:
            self.cancel(handle.request.workspace_id, handle.request.conversation_id, handle.request.turn_id)
        await asyncio.gather(*(self.wait(handle) for handle in handles), return_exceptions=True)
        await asyncio.gather(*list(self._deletions.values()), return_exceptions=True)
