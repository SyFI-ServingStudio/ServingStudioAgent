"""Best-effort names released only after their source turn commits."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from ..domain.turns import Outcome, TurnInput, TurnResult
from ..storage.conversations import Conversations
from ..storage.registry import WorkspaceRegistry
from .name_generator import NameGenerator


@dataclass(frozen=True)
class NamingTicket:
    ready: asyncio.Event
    task: asyncio.Task

    def commit(self) -> None:
        self.ready.set()

    def abort(self) -> None:
        self.task.cancel()


class NamingService:
    def __init__(
        self,
        registry: WorkspaceRegistry,
        conversations: Callable[[str], Conversations],
        generator: NameGenerator,
        *,
        logger: logging.Logger | None = None,
    ):
        self.registry = registry
        self.conversations = conversations
        self.generator = generator
        self.logger = logger or logging.getLogger("vibesim_agent.naming")
        self._pending: dict[tuple[str, str], NamingTicket] = {}
        self._closing = False
        self._close_task: asyncio.Task | None = None

    def prepare(self, request: TurnInput, result: TurnResult) -> NamingTicket | None:
        if (
            self._closing
            or not self.generator.enabled
            or result.outcome not in {Outcome.ANSWER, Outcome.INPUT}
            or not result.text.strip()
            or result.text == "(no answer)"
            or result.metadata.get("failure") is not None
        ):
            return None
        key = (request.workspace_id, request.conversation_id)
        if key in self._pending:
            return None
        try:
            conversation = self.conversations(key[0]).get(key[1])
            if conversation is None:
                return None
            workspace = self.registry.get(key[0])
            if (
                workspace["naming_state"] != "pending"
                and conversation["naming_state"] != "pending"
            ):
                return None
            ready = asyncio.Event()
            operation = self._run(key, ready, request.text, result.text)
            try:
                task = asyncio.create_task(operation, name=f"naming-{request.turn_id}")
            except BaseException:
                operation.close()
                raise
            ticket = NamingTicket(ready, task)
            self._pending[key] = ticket

            def finished(task):
                if self._pending.get(key) is ticket:
                    del self._pending[key]
                if not task.cancelled():
                    task.exception()

            task.add_done_callback(finished)
            return ticket
        except Exception:  # noqa: BLE001 - Optional naming must not fail a turn.
            self.logger.warning(
                "Automatic naming could not be scheduled",
                extra={"workspace_id": key[0], "conversation_id": key[1]},
            )
            return None

    async def _run(self, key, ready, user_message, final_answer) -> None:
        await ready.wait()
        try:
            names = await self.generator.generate(user_message, final_answer)
            conversations = self.conversations(key[0])
            # Deletion while the model is running must not rename the workspace
            # on behalf of a conversation that no longer exists.
            if conversations.get(key[1]) is None:
                return
            self.registry.apply_generated_name(key[0], names.workspace_name)
            conversations.apply_generated_title(key[1], names.conversation_title)
        except Exception as error:  # noqa: BLE001 - Keep failures isolated from chat.
            # Keep pending state for a later successful turn. Provider exceptions
            # may embed request headers or generated text, so only their type is
            # logged: enough to tell a timeout from an empty or invalid reply.
            self.logger.warning(
                "Automatic naming failed (%s)",
                type(error).__name__,
                extra={"workspace_id": key[0], "conversation_id": key[1]},
            )

    def stop_accepting(self) -> None:
        self._closing = True

    async def close(self) -> None:
        self.stop_accepting()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._drain())
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _drain(self) -> None:
        tickets = list(self._pending.values())
        for ticket in tickets:
            if not ticket.ready.is_set():
                ticket.abort()
        await asyncio.gather(
            *(ticket.task for ticket in tickets), return_exceptions=True
        )
