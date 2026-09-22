"""The workspace kind axis and the execution mode derived from it."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class WorkspaceKind(StrEnum):
    """What the user asked for, which `storage_kind` alone cannot express.

    `w_main` and a provisioned worktree are both `external` and both execute the
    same way, but they are not the same thing to a user: one is the shared
    checkout everyone starts in, the other is a branch this workspace owns.
    """

    COPY = "copy"
    WORKTREE = "worktree"
    CHECKOUT = "checkout"


class ExecutionMode(StrEnum):
    CONTAINER = "container"
    HOST = "host"


def workspace_kind(descriptor: dict[str, Any]) -> WorkspaceKind:
    """Descriptors written before the kind axis existed still have to answer."""
    recorded = descriptor.get("workspace_kind")
    if recorded in tuple(WorkspaceKind):
        return WorkspaceKind(recorded)
    if descriptor.get("storage_kind") == "managed":
        return WorkspaceKind.COPY
    return WorkspaceKind.CHECKOUT


def execution_mode(descriptor: dict[str, Any]) -> ExecutionMode:
    """Single source of truth for the mode; the runtime branches on this too.

    Reported rather than stored. A stored copy would be a second answer that
    goes stale the moment the mapping changes, and it is about to change once
    host execution is switched on.
    """
    del descriptor
    return ExecutionMode.CONTAINER
