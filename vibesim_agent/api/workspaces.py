"""Workspace browser and tools endpoints share explicit provisioning."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from ..domain.workspaces import WorkspaceKind, execution_mode, workspace_kind
from ..runtime.worktree import WorktreeError
from ..services.workspace import WorkspaceService, WorktreeUnavailable


def described(descriptor: dict) -> dict:
    """Answer the kind and execution questions here, once.

    Descriptors written before the kind axis exist -- `w_main` among them -- and
    execution is derived rather than stored. Backfilling at the boundary keeps
    the UI from carrying a second copy of both rules that can drift from this
    one.
    """
    return {
        **descriptor,
        "workspace_kind": workspace_kind(descriptor).value,
        "execution": execution_mode(descriptor).value,
    }


class NewWorkspace(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    display_name: str = Field(alias="displayName")
    auto_name: bool = Field(default=False, alias="autoName")
    # An intent enum, not `storageKind`: that is registry state, and "external"
    # in a *request* conflates "make me a worktree" with the far more dangerous
    # "adopt this path I am handing you". `kind` leaves room for the latter.
    kind: WorkspaceKind = WorkspaceKind.COPY
    branch: str | None = None
    base: str | None = None


class UpdateWorkspace(BaseModel):
    display_name: str | None = Field(default=None, alias="displayName")
    state: str | None = None


BranchTopic = Callable[[str], Awaitable[str]]
logger = logging.getLogger("vibesim_agent.workspaces")


async def _proposed_topic(branch_topic: BranchTopic | None, request: str) -> str | None:
    """The naming model's branch, or None to fall back to the display name.

    Never an error: a branch name is not worth failing a workspace over, and
    the fallback is the name this endpoint always used.
    """
    if branch_topic is None:
        return None
    try:
        return await branch_topic(request)
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as error:  # noqa: BLE001 - any failure means "use the fallback"
        logger.info("branch naming unavailable, using the display name: %s", error)
        return None


def workspace_collection_router(
    service: WorkspaceService,
    *,
    prefix: str = "/api/agent/v1/workspaces",
    branch_topic: BranchTopic | None = None,
) -> APIRouter:
    router = APIRouter(prefix=prefix)

    @router.get("")
    def list_workspaces():
        # The capability key ships with the endpoint, not after it. Without it
        # the UI cannot tell an old server from the feature being switched off,
        # and those need different treatment: hide the row, or explain why.
        return {
            "workspaces": [described(item) for item in service.registry.list()],
            "capabilities": {
                "workspaceKinds": [kind.value for kind in service.workspace_kinds]
            },
        }

    @router.post("")
    async def create_workspace(body: NewWorkspace):
        naming_state = "pending" if body.auto_name else "manual"
        try:
            if body.kind is WorkspaceKind.WORKTREE:
                # Named before it exists, because a branch cannot be renamed
                # afterwards the way the display name is. A branch the reader
                # typed is theirs and is never replaced.
                topic = (
                    await _proposed_topic(branch_topic, body.display_name)
                    if body.branch is None
                    else None
                )
                return described(
                    await run_in_threadpool(
                        service.create_worktree,
                        body.display_name,
                        branch=body.branch,
                        base=body.base,
                        naming_state=naming_state,
                        topic=topic,
                    )
                )
            if body.branch is not None or body.base is not None:
                raise ValueError("branch and base apply to worktree workspaces only")
            return described(
                await run_in_threadpool(
                    service.create, body.display_name, naming_state=naming_state
                )
            )
        except WorktreeUnavailable as error:
            raise HTTPException(501, str(error)) from None
        except FileExistsError as error:
            raise HTTPException(409, str(error)) from None
        except (ValueError, WorktreeError) as error:
            raise HTTPException(400, str(error)) from None

    return router


def workspace_router(
    service: WorkspaceService, *, branch_topic: BranchTopic | None = None
) -> APIRouter:
    router = workspace_collection_router(service, branch_topic=branch_topic)

    @router.get("/{workspace_id}")
    def get_workspace(workspace_id: str):
        try:
            return described(service.registry.get(workspace_id))
        except (KeyError, ValueError):
            raise HTTPException(404, "workspace not found") from None

    @router.patch("/{workspace_id}")
    def update_workspace(workspace_id: str, body: UpdateWorkspace):
        try:
            return described(
                service.registry.update(
                    workspace_id, display_name=body.display_name, state=body.state
                )
            )
        except KeyError:
            raise HTTPException(404, "workspace not found") from None
        except ValueError as error:
            raise HTTPException(400, str(error)) from None

    return router
