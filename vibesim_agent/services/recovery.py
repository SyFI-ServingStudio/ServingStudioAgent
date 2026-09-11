"""Recover pre-start turns while holding exclusive ownership of workspace state."""

from collections.abc import Awaitable, Callable

from ..storage.ownership import WorkspaceOwnership
from ..storage.registry import WorkspaceRegistry
from .capabilities import CapabilityRegistry, ManagedContext
from .turn import TurnStorage, history_activity

BACKEND_RESTART_MESSAGE = (
    "This turn was interrupted when the conversation backend restarted. "
    "Completed Agent activity was recovered above, but no final answer was produced. "
    "Send `continue` to resume from the existing workspace and role sessions."
)


class RecoveryService:
    def __init__(
        self,
        workspaces: WorkspaceRegistry,
        storage: Callable[[str], TurnStorage],
        *,
        capabilities: CapabilityRegistry,
        context: ManagedContext,
        remove_container: Callable[[str, str], Awaitable[None]],
        ownership: WorkspaceOwnership | None = None,
    ):
        self.workspaces = workspaces
        self.storage = storage
        self.capabilities = capabilities
        self.context = context
        self.remove_container = remove_container
        self.ownership = ownership or WorkspaceOwnership(workspaces.root)
        if self.ownership.root != workspaces.root:
            raise ValueError("recovery ownership must match workspace state root")
        self._recovering = False
        self._ready = False

    async def recover(self) -> int:
        if self._ready:
            return 0
        if self._recovering:
            raise RuntimeError("startup recovery is already in progress")
        # Lock the stable registry directory, so another backend cannot recover
        # this process's live turns. Keep ownership through shutdown and drain.
        self.ownership.acquire()
        self._recovering = True
        try:
            # Validate every database before any cleanup or recovery writes.
            pending = []
            for workspace in self.workspaces.list(include_archived=True):
                workspace_id = workspace["workspace_id"]
                store = self.storage(workspace_id)
                pending.append((workspace_id, store, store.turns.running()))
            recovered = 0
            for workspace_id, store, turns in pending:
                for turn in turns:
                    conversation_id, turn_id = turn["conversation_id"], turn["id"]
                    self.capabilities.revoke_turn(workspace_id, turn_id)
                    await self.remove_container(workspace_id, conversation_id)
                    self.context.remove(workspace_id, conversation_id)
                    activity, _ = history_activity(
                        store.turns.events(conversation_id, turn_id), workspace_id
                    )
                    activity.append({"kind": "error", "text": BACKEND_RESTART_MESSAGE})
                    recovered += store.turns.interrupt(
                        conversation_id,
                        turn_id,
                        text=BACKEND_RESTART_MESSAGE,
                        metadata={"activity": activity},
                    )
            self._ready = True
            return recovered
        except BaseException:
            self.close()
            raise
        finally:
            self._recovering = False

    def close(self) -> None:
        self._ready = False
        self.ownership.close()
