"""Synchronous Agent presentation over the same durable background turns."""

from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from starlette.responses import FileResponse

from ..domain.errors import ProviderUnavailable
from ..domain.evidence import AnalyzerTurnContext
from ..domain.results import project_turn_result
from ..domain.roles import AgentMode, Sandbox
from ..domain.turns import TurnOptions
from ..services.artifacts import ArtifactService
from ..services.conversation import ConversationService
from ..services.eval import EvalService
from ..services.turn import TurnService
from ..services.workspace import WorkspaceService
from .conversations import conversation_collection_router, delete_conversation
from .deps import token_dependency
from .files import artifact_router
from .workspaces import workspace_collection_router


class SendToolMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    text: str
    sandbox_mode: Sandbox | None = None
    autonomous_mode: bool | None = None
    agent_mode: str | None = Field(default=None, alias="agentMode")
    analyzer_context: AnalyzerTurnContext | None = Field(
        default=None, alias="analyzerContext"
    )


class EvalRequest(BaseModel):
    prompt: str
    sandbox: Sandbox = Sandbox.WORKSPACE_WRITE
    autonomous: bool = True
    agent_mode: str = AgentMode.ORCHESTRATED.value
    keep_container: bool = False


def tools_router(
    turns: TurnService,
    conversations: ConversationService,
    *,
    token: SecretStr,
    skill_document: Path,
    prepare_workspace: Callable[[str], str] | None = None,
    cleanup_conversation: Callable[[str, str], Awaitable[None]] | None = None,
    workspaces: WorkspaceService | None = None,
    evaluations: EvalService | None = None,
    artifacts: ArtifactService | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/agent/v1/tools")
    protected = APIRouter(dependencies=[Depends(token_dependency(token))])
    if artifacts is not None:
        protected.include_router(artifact_router(artifacts))
    if evaluations is not None:

        @protected.post("/eval")
        async def evaluate(body: EvalRequest):
            if body.agent_mode not in {mode.value for mode in AgentMode}:
                raise HTTPException(
                    400, {"code": "unknown_agent_mode", "agent_mode": body.agent_mode}
                )
            try:
                return await evaluations.run(
                    body.prompt,
                    sandbox=body.sandbox,
                    autonomous=body.autonomous,
                    mode=AgentMode(body.agent_mode),
                    keep_container=body.keep_container,
                )
            except ProviderUnavailable as error:
                raise HTTPException(
                    409,
                    {
                        "code": "codex_family_unavailable",
                        "families": sorted(error.provider_ids),
                    },
                ) from None
            except ValueError as error:
                raise HTTPException(400, str(error)) from None

    if workspaces is not None:
        protected.include_router(
            workspace_collection_router(workspaces, prefix="/workspaces")
        )
    protected.include_router(
        conversation_collection_router(
            conversations,
            prepare_workspace=prepare_workspace,
            prefix="/workspaces/{workspace_id}/conversations",
        )
    )

    @router.get("/skill")
    async def skill():
        if not skill_document.is_file():
            raise HTTPException(404, "SKILL.md not found")
        return FileResponse(skill_document, media_type="text/markdown")

    @protected.get("/workspaces/{workspace_id}/conversations/{cid}")
    async def get_conversation(workspace_id: str, cid: str):
        try:
            return conversations.get(workspace_id, cid)
        except (KeyError, ValueError):
            raise HTTPException(404, "conversation not found") from None

    @protected.delete("/workspaces/{workspace_id}/conversations/{cid}")
    async def remove(workspace_id: str, cid: str):
        return await delete_conversation(turns, cleanup_conversation, workspace_id, cid)

    @protected.post("/workspaces/{workspace_id}/conversations/{cid}/messages")
    async def send(workspace_id: str, cid: str, body: SendToolMessage):
        try:
            store = turns.storage(workspace_id)
        except (KeyError, ValueError):
            raise HTTPException(404, "workspace not found") from None
        if store.conversations.get(cid) is None:
            raise HTTPException(404, "conversation not found")
        if not body.text.strip():
            raise HTTPException(400, "empty message")
        if body.agent_mode is not None and body.agent_mode not in {
            mode.value for mode in AgentMode
        }:
            raise HTTPException(
                400, {"code": "unknown_agent_mode", "agent_mode": body.agent_mode}
            )
        try:
            handle = turns.start(
                workspace_id,
                cid,
                body.text,
                options=TurnOptions(
                    sandbox=body.sandbox_mode,
                    autonomous=body.autonomous_mode,
                    mode=AgentMode(body.agent_mode)
                    if body.agent_mode is not None
                    else None,
                    analyzer_context=body.analyzer_context.model_dump(by_alias=True)
                    if body.analyzer_context is not None
                    else None,
                ),
            )
        except ProviderUnavailable as error:
            raise HTTPException(
                409,
                {
                    "code": "codex_family_unavailable",
                    "families": sorted(error.provider_ids),
                },
            ) from None
        except ValueError as error:
            raise HTTPException(409, str(error)) from None
        result = await turns.wait(handle)
        projected = project_turn_result(
            handle.request, result, store.turns.events(cid, handle.request.turn_id)
        )
        return {
            **projected,
            "citations": result.metadata.get("citations", []),
            "citation_dictionary_id": result.metadata.get("citation_dictionary_id"),
            "failure": result.metadata.get("failure"),
            "naming_scheduled": result.metadata.get("naming_scheduled", False),
        }

    router.include_router(protected)
    return router
