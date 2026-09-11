"""Workspace browser and tools endpoints share explicit provisioning."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..services.workspace import WorkspaceService


class NewWorkspace(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    display_name: str = Field(alias="displayName")
    auto_name: bool = Field(default=False, alias="autoName")


class UpdateWorkspace(BaseModel):
    display_name: str | None = Field(default=None, alias="displayName")
    state: str | None = None


def workspace_collection_router(
    service: WorkspaceService, *, prefix: str = "/api/agent/v1/workspaces"
) -> APIRouter:
    router = APIRouter(prefix=prefix)

    @router.get("")
    def list_workspaces():
        return {"workspaces": service.registry.list()}

    @router.post("")
    def create_workspace(body: NewWorkspace):
        try:
            return service.create(
                body.display_name,
                naming_state="pending" if body.auto_name else "manual",
            )
        except ValueError as error:
            raise HTTPException(400, str(error)) from None

    return router


def workspace_router(service: WorkspaceService) -> APIRouter:
    router = workspace_collection_router(service)

    @router.get("/{workspace_id}")
    def get_workspace(workspace_id: str):
        try:
            return service.registry.get(workspace_id)
        except (KeyError, ValueError):
            raise HTTPException(404, "workspace not found") from None

    @router.patch("/{workspace_id}")
    def update_workspace(workspace_id: str, body: UpdateWorkspace):
        try:
            return service.registry.update(
                workspace_id, display_name=body.display_name, state=body.state
            )
        except KeyError:
            raise HTTPException(404, "workspace not found") from None
        except ValueError as error:
            raise HTTPException(400, str(error)) from None

    return router
