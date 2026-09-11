"""MCP citation registration using the same per-turn authority as Launcher."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..services.capabilities import Capability, CapabilityRegistry
from ..services.citations import CitationConflict, CitationService
from .deps import capability_dependency


class RegisterCitations(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    resource_kind: Literal[
        "aggregate", "run", "prediction", "kernel_profile", "kernel_measurement"
    ] = Field(default="aggregate", alias="resourceKind")
    experiment_id: str | None = Field(default=None, alias="experimentId")
    prediction_id: str | None = Field(default=None, alias="predictionId")
    run_id: str | None = Field(default=None, alias="runId")
    profile_id: str | None = Field(default=None, alias="profileId")
    measurement_id: str | None = Field(default=None, alias="measurementId")
    resource_path: str | None = Field(default=None, alias="resourcePath")
    analysis: dict | None = None


def citation_router(
    service: CitationService, capabilities: CapabilityRegistry
) -> APIRouter:
    router = APIRouter(prefix="/api/agent/v1/internal/analyzer-citations")
    require_capability = capability_dependency(capabilities)

    @router.post("/register")
    async def register(
        body: RegisterCitations,
        capability: Annotated[Capability, Depends(require_capability)],
    ):
        try:
            return service.register(capability, **body.model_dump())
        except KeyError as error:
            raise HTTPException(404, str(error.args[0])) from None
        except CitationConflict as error:
            raise HTTPException(409, str(error)) from None
        except (TypeError, ValueError) as error:
            raise HTTPException(400, str(error)) from None

    return router
