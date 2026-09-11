"""Capability-gated Launcher callbacks and browser ownership overlays."""

from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..services.capabilities import Capability, CapabilityRegistry
from ..services.jobs import JobConflict, JobService
from .deps import capability_dependency


class RegisterRun(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    experiment_root: str = Field(alias="experimentRoot")
    run_count: int = Field(default=1, ge=1, alias="runCount")
    axes: list[str] = Field(default_factory=list)


class RegisterJob(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    job_kind: str = Field(alias="jobKind", min_length=1)
    artifact_root: str = Field(alias="artifactRoot", min_length=1)
    analyzer_resource_id: str | None = Field(default=None, alias="analyzerResourceId")


class UpdateJob(BaseModel):
    status: str


class RegisterUnifiedJob(RegisterJob):
    job_kind: Literal[
        "simulation", "timing_predict", "kernel_profile", "kernel_measure"
    ] = Field(alias="jobKind")
    run_count: int = Field(default=1, ge=1, alias="runCount")
    axes: list[str] = Field(default_factory=list)


def job_router(jobs: JobService, capabilities: CapabilityRegistry) -> APIRouter:
    router = APIRouter(prefix="/api/agent/v1")

    require_capability = capability_dependency(capabilities)

    def call(operation: Callable, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except KeyError as error:
            raise HTTPException(404, str(error.args[0])) from None
        except JobConflict as error:
            raise HTTPException(409, str(error)) from None
        except PermissionError as error:
            raise HTTPException(403, str(error)) from None
        except ValueError as error:
            raise HTTPException(400, str(error)) from None

    @router.post("/internal/jobs/register")
    async def register(
        body: RegisterUnifiedJob,
        capability: Annotated[Capability, Depends(require_capability)],
    ):
        return call(jobs.register, capability, **body.model_dump())

    @router.post("/internal/jobs/{job_id}/status")
    async def update(
        job_id: str,
        body: UpdateJob,
        capability: Annotated[Capability, Depends(require_capability)],
    ):
        return call(jobs.update_registered, capability, job_id, body.status)

    @router.post("/internal/managed-runs/register")
    async def register_run(
        body: RegisterRun,
        capability: Annotated[Capability, Depends(require_capability)],
    ):
        return call(jobs.register_run, capability, **body.model_dump())

    @router.post("/internal/managed-jobs/register")
    async def register_job(
        body: RegisterJob,
        capability: Annotated[Capability, Depends(require_capability)],
    ):
        return call(jobs.register_artifact, capability, **body.model_dump())

    @router.post("/internal/managed-runs/{job_id}/status")
    async def update_run(
        job_id: str,
        body: UpdateJob,
        capability: Annotated[Capability, Depends(require_capability)],
    ):
        return call(jobs.update, capability, job_id, body.status, simulation=True)

    @router.post("/internal/managed-jobs/{job_id}/status")
    async def update_job(
        job_id: str,
        body: UpdateJob,
        capability: Annotated[Capability, Depends(require_capability)],
    ):
        return call(jobs.update, capability, job_id, body.status, simulation=False)

    @router.get("/jobs")
    async def list_jobs():
        return {"jobs": call(jobs.list_jobs)}

    @router.get("/workspaces/{workspace_id}/jobs/{resource_id}")
    async def get_resource(workspace_id: str, resource_id: str):
        return call(jobs.resource, workspace_id, resource_id)

    @router.get("/workspaces/{workspace_id}/conversations/{cid}/experiments")
    async def list_experiments(workspace_id: str, cid: str):
        return call(jobs.experiments, workspace_id, cid)

    combined = APIRouter()
    combined.include_router(router)
    for path, endpoint in (
        ("managed-runs/register", register_run),
        ("managed-jobs/register", register_job),
        ("managed-runs/{job_id}/status", update_run),
        ("managed-jobs/{job_id}/status", update_job),
    ):
        combined.add_api_route(
            f"/api/internal/{path}", endpoint, methods=["POST"], include_in_schema=False
        )
    return combined
