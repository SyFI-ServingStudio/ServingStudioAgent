"""Browser chat routes; turn execution and cancellation live in the service."""

import json
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response, StreamingResponse

from ..domain.errors import ProviderUnavailable
from ..domain.events import MANAGED_JOB_EVENTS
from ..domain.evidence import AnalyzerTurnContext
from ..domain.roles import AgentMode, Role, Sandbox
from ..domain.turns import TurnOptions
from ..services.conversation import (
    ConversationRuntimeLocked,
    ConversationService,
    UnknownConversationModel,
)
from ..services.turn import TurnService


class SendMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    text: str
    sandbox_mode: Sandbox = Sandbox.WORKSPACE_WRITE
    autonomous_mode: bool = False
    agent_mode: str | None = Field(default=None, alias="agentMode")
    analyzer_context: AnalyzerTurnContext | None = Field(
        default=None, alias="analyzerContext"
    )
    resume_role: Role | Literal[""] | None = Field(default=None, alias="resumeRole")


class RuntimeOverride(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    provider: str | None = Field(default=None, min_length=1)
    model: str = ""
    effort: str = ""
    service_tier: str = Field(default="", alias="serviceTier")


class ConversationRuntime(BaseModel):
    orchestrator: RuntimeOverride = Field(default_factory=RuntimeOverride)
    implementer: RuntimeOverride = Field(default_factory=RuntimeOverride)
    assistant: RuntimeOverride = Field(default_factory=RuntimeOverride)


class UpdateConversationRuntime(BaseModel):
    codex_runtime: ConversationRuntime


class RenameConversation(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def conversation_index_router(conversations: ConversationService) -> APIRouter:
    router = APIRouter()

    @router.get("/api/agent/v1/conversations")
    async def list_all():
        return {
            "conversations": conversations.list_all(),
            "sandbox_modes": [sandbox.value for sandbox in Sandbox],
        }

    return router


class NewConversation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    sandbox: Sandbox = Sandbox.WORKSPACE_WRITE
    autonomous: bool = False
    agent_mode: str = Field(default=AgentMode.ORCHESTRATED.value, alias="agentMode")
    codex_runtime: ConversationRuntime = Field(default_factory=ConversationRuntime)
    peer_workspace: str | None = None
    eager: bool = False


def conversation_collection_router(
    conversations: ConversationService,
    *,
    prepare_workspace: Callable[[str], str] | None = None,
    prefix: str = "/api/agent/v1/workspaces/{workspace_id}/conversations",
) -> APIRouter:
    router = APIRouter(prefix=prefix)

    @router.get("")
    async def list_conversations(workspace_id: str):
        try:
            records = conversations.list(workspace_id)
        except (KeyError, ValueError):
            raise HTTPException(404, "workspace not found") from None
        return {
            "workspace_id": workspace_id,
            "conversations": records,
            "sandbox_modes": [sandbox.value for sandbox in Sandbox],
        }

    @router.post("")
    async def create_conversation(workspace_id: str, body: NewConversation):
        try:
            conversations.storage(workspace_id)
        except (KeyError, ValueError):
            raise HTTPException(404, "workspace not found") from None
        if body.eager and prepare_workspace is None:
            raise HTTPException(409, "eager workspace preparation is not configured")
        try:
            runtimes = conversations.browser_runtimes(
                body.codex_runtime.model_dump(exclude_unset=True)
            )
            if body.agent_mode not in {mode.value for mode in AgentMode}:
                raise HTTPException(
                    400, {"code": "unknown_agent_mode", "agent_mode": body.agent_mode}
                )
            result = conversations.create(
                workspace_id,
                mode=AgentMode(body.agent_mode),
                sandbox=body.sandbox,
                autonomous=body.autonomous,
                peer_workspace=body.peer_workspace,
                runtimes=runtimes,
            )
        except UnknownConversationModel as error:
            raise HTTPException(
                400, {"code": "unknown_codex_model", "model": error.model_id}
            ) from None
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
        if body.eager:
            result["workspace_path"] = str(
                await run_in_threadpool(prepare_workspace, workspace_id)
            )
        return result

    return router


def browser_event(kind: str, payload: dict) -> dict:
    if kind in MANAGED_JOB_EVENTS:
        return dict(payload)
    if kind in {"role_start", "role_ready"}:
        return {"role": payload.get("role")}
    if kind == "session":
        return {key: payload.get(key) for key in ("role", "backend", "session_id")}
    if kind in {"tool_call", "error", "implementer"}:
        return {"text": payload.get("text", "")}
    data = {key: value for key, value in payload.items() if key != "kind"}
    if kind == "done":
        data.setdefault("failure", None)
        data.setdefault("naming_scheduled", False)
    return data


async def delete_conversation(
    turns: TurnService, cleanup, workspace_id: str, conversation_id: str
) -> dict:
    if cleanup is None:
        raise HTTPException(409, "conversation cleanup is not configured")
    try:
        await turns.delete(workspace_id, conversation_id, cleanup)
    except KeyError:
        raise HTTPException(404, "workspace not found") from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    return {"ok": True}


def conversation_router(
    turns: TurnService,
    queries: ConversationService,
    *,
    cleanup_conversation: Callable[[str, str], Awaitable[None]] | None = None,
) -> APIRouter:
    router = APIRouter(
        prefix="/api/agent/v1/workspaces/{workspace_id}/conversations/{cid}"
    )

    @router.patch("/runtime")
    async def update_runtime(
        workspace_id: str, cid: str, body: UpdateConversationRuntime
    ):
        try:
            return queries.update_runtime(
                workspace_id, cid, body.codex_runtime.model_dump(exclude_unset=True)
            )
        except UnknownConversationModel as error:
            raise HTTPException(
                400, {"code": "unknown_codex_model", "model": error.model_id}
            ) from None
        except ProviderUnavailable as error:
            raise HTTPException(
                409,
                {
                    "code": "codex_family_unavailable",
                    "families": sorted(error.provider_ids),
                },
            ) from None
        except ConversationRuntimeLocked as error:
            raise HTTPException(
                409, {"code": "conversation_runtime_locked", "message": str(error)}
            ) from None
        except KeyError:
            raise HTTPException(404, "conversation not found") from None
        except ValueError as error:
            raise HTTPException(400, str(error)) from None

    @router.patch("")
    async def rename(workspace_id: str, cid: str, body: RenameConversation):
        try:
            return queries.rename(workspace_id, cid, body.title)
        except KeyError:
            raise HTTPException(404, "conversation not found") from None
        except ValueError as error:
            raise HTTPException(400, str(error)) from None

    @router.delete("")
    async def remove(workspace_id: str, cid: str):
        return await delete_conversation(turns, cleanup_conversation, workspace_id, cid)

    def require_conversation(workspace_id, cid):
        try:
            store = turns.storage(workspace_id)
        except KeyError:
            raise HTTPException(404, "workspace not found") from None
        if store.conversations.get(cid) is None:
            raise HTTPException(404, "conversation not found")

    def stream_response(workspace_id, cid, turn_id):
        async def stream():
            async with aclosing(turns.stream(workspace_id, cid, turn_id)) as events:
                async for event in events:
                    payload = browser_event(event["kind"], event["payload"])
                    kind = (
                        "job" if event["kind"] in MANAGED_JOB_EVENTS else event["kind"]
                    )
                    yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "X-Turn-Id": turn_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("")
    async def history(
        workspace_id: str,
        cid: str,
        limit: int | None = Query(default=None, ge=1, le=100),
        before: int | None = Query(default=None, ge=0),
    ):
        try:
            return queries.get(workspace_id, cid, limit=limit, before=before)
        except KeyError:
            raise HTTPException(404, "conversation not found") from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @router.post("/messages")
    async def send(workspace_id: str, cid: str, body: SendMessage):
        require_conversation(workspace_id, cid)
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
                    resume_role=body.resume_role,
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
        return stream_response(workspace_id, cid, handle.request.turn_id)

    @router.get("/stream")
    async def reconnect(workspace_id: str, cid: str):
        require_conversation(workspace_id, cid)
        handle = turns.current(workspace_id, cid)
        if handle is None:
            if turns.storage(workspace_id).turns.active(cid) is not None:
                raise HTTPException(409, "turn requires recovery")
            return Response(status_code=204)
        return stream_response(workspace_id, cid, handle.request.turn_id)

    @router.get("/turns")
    async def turn_index(workspace_id: str, cid: str):
        require_conversation(workspace_id, cid)
        return {"turns": turns.storage(workspace_id).turns.list(cid)}

    @router.get("/turns/{turn_id}/replay")
    async def replay(workspace_id: str, cid: str, turn_id: str):
        require_conversation(workspace_id, cid)
        store = turns.storage(workspace_id)
        if store.turns.get(cid, turn_id) is None:
            raise HTTPException(404, "turn not found")
        records = store.turns.events(cid, turn_id)

        async def stream():
            for event in records:
                yield f"event: {event['kind']}\ndata: {json.dumps(event['payload'], ensure_ascii=False)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/cancel")
    async def cancel(workspace_id: str, cid: str, turn_id: str | None = None):
        require_conversation(workspace_id, cid)
        handle = turns.current(workspace_id, cid)
        if handle is None:
            return {"cancelled": False}
        if turn_id is not None and turn_id != handle.request.turn_id:
            return {"cancelled": False, "stale": True}
        turns.cancel(workspace_id, cid, handle.request.turn_id)
        result = await turns.wait(handle)
        return {
            "cancelled": result.outcome.value == "cancelled",
            "interrupted_role": result.resume_role.value
            if result.resume_role is not None
            else "",
        }

    return router
