"""Durable evaluations whose preparation and execution outlive HTTP waiters."""

import asyncio
from collections.abc import Awaitable, Callable
from uuid import uuid4

from ..domain.results import project_turn_result
from ..domain.roles import AgentMode, Sandbox
from ..domain.turns import TurnInput
from .conversation import ConversationService
from .turn import TurnService
from .workspace import WorkspaceService


class EvalService:
    def __init__(
        self,
        workspaces: WorkspaceService,
        conversations: ConversationService,
        turns: TurnService,
        *,
        release: Callable[[str, str], Awaitable[None]],
    ):
        self.workspaces = workspaces
        self.conversations = conversations
        self.turns = turns
        self.release = release
        self._tasks: set[asyncio.Task] = set()
        self._closing = False
        self._close_task: asyncio.Task | None = None

    async def run(
        self,
        prompt: str,
        *,
        sandbox: Sandbox = Sandbox.WORKSPACE_WRITE,
        autonomous: bool = True,
        mode: AgentMode = AgentMode.ORCHESTRATED,
        keep_container: bool = False,
    ) -> dict:
        if self._closing:
            raise RuntimeError("evaluation service is shutting down")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("empty prompt")
        sandbox, mode = Sandbox(sandbox), AgentMode(mode)
        task = asyncio.create_task(
            self._run(prompt, sandbox, autonomous, mode, keep_container),
            name="agent-evaluation",
        )
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        return await asyncio.shield(task)

    def _finished(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _run(self, prompt, sandbox, autonomous, mode, keep_container) -> dict:
        conversation_id = f"eval-{uuid4().hex[:12]}"
        workspace_id = f"w_{conversation_id}"
        await asyncio.to_thread(
            self.workspaces.create,
            f"Evaluation {conversation_id[-6:]}",
            workspace_id=workspace_id,
        )
        # close() drains preparation instead of cancelling its filesystem thread.
        if self._closing:
            raise RuntimeError("evaluation service closed during workspace preparation")
        self.conversations.create(
            workspace_id,
            conversation_id=conversation_id,
            mode=mode,
            sandbox=sandbox,
            autonomous=autonomous,
            peer_workspace=None,
        )
        retained = True

        async def cleanup(request: TurnInput) -> None:
            nonlocal retained
            await self.release(request.workspace_id, request.conversation_id)
            retained = False

        handle = self.turns.start(
            workspace_id,
            conversation_id,
            prompt,
            after_execution=None if keep_container else cleanup,
            auto_name=False,
        )
        result = await self.turns.wait(handle)
        events = self.turns.storage(workspace_id).turns.events(
            conversation_id, handle.request.turn_id
        )
        return {
            **project_turn_result(handle.request, result, events),
            "workspace_id": workspace_id,
            "workspace": str(self.workspaces.registry.repo_path(workspace_id)),
            "kept_workspace": True,
            "kept_container": retained,
        }

    async def close(self) -> None:
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._shutdown())
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _shutdown(self) -> None:
        await self.turns.close()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
