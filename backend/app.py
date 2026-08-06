"""FastAPI app: REST + SSE chat over Docker-backed `codex exec`.

Endpoints:
  GET    /                              -> Vite frontend index
  GET    /assets/*                      -> Vite frontend assets
  GET    /api/conversations             -> standalone shell compatibility index
  GET/POST /api/workspaces              -> list/create durable workspaces
  GET/PATCH /api/workspaces/{wid}       -> workspace descriptor
  GET/POST /api/workspaces/{wid}/conversations -> list/create conversations
  GET/DELETE /api/workspaces/{wid}/conversations/{cid} -> history/delete
  POST   /api/workspaces/{wid}/conversations/{cid}/messages -> SSE turn
  GET    /api/workspaces/{wid}/conversations/{cid}/stream -> reconnect
  POST   /api/workspaces/{wid}/conversations/{cid}/cancel -> cancel
  GET    /api/workspaces/{wid}/conversations/{cid}/experiments -> linked results
  GET    /api/file, /api/file/meta, /api/file/list -> workspace file preview
  POST   /api/internal/managed-runs/*   -> simulation compatibility callbacks
  POST   /api/internal/managed-jobs/*   -> typed capability-gated job callbacks
  POST   /api/eval                          -> JSON single-turn eval (evaluation only)
  GET    /api/agent/skill                    -> agent skill doc (SKILL.md, public)
  GET    /api/agent/workspaces/{wid}/artifacts -> list workspace artifacts
  POST   /api/agent/workspaces/{wid}/conversations -> create agent conversation
  POST   /api/agent/workspaces/{wid}/conversations/{cid}/messages -> sync turn

All agent-facing endpoints share the /api/agent/* prefix and are gated by
`require_token` when VIBESIM_API_TOKEN is set; /api/agent/skill stays public so an
agent can learn the contract before it holds a token. The
`/api/agent/workspaces/*/conversations*` endpoints are the real interactive interface
(multi-turn, session + workspace continuity, reusing `store` and `run_turn`);
`/api/eval` is single-turn and evaluation-only (kept outside the /api/agent/*
namespace on purpose). The browser SSE endpoints (/api/workspaces/*/conversations*) are
unchanged and not token-gated.

Browser turns run independently of any one HTTP connection, so a refreshed page
can replay and continue the active SSE stream. The message endpoint streams
Server-Sent Events: `session` (role Codex session id),
`tool_call` (transient command/tool activity), `intermediate_output` (assistant
progress or milestone commentary),
`decision` (orchestrator→implementer delegated task), `usage` (per-call duration +
token breakdown), `implementer` (implementer summary), then `done` (stored final
answer). `implementer`/`decision`/`usage` can repeat inside one browser turn when the
orchestrator issues follow-up tasks. The render-relevant events are also persisted as
an ordered `activity` list on the assistant message so a reload rebuilds the same role
timeline. A per-conversation lock prevents two turns racing the same Codex
session/container; conversations in one workspace intentionally share its repo
and experiment state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .analyzer_context import (
    AnalyzerTurnContext,
    CitationDictionarySnapshot,
    build_aggregate_citation_dictionary,
    build_kernel_measurement_citation_dictionary,
    build_kernel_profile_citation_dictionary,
    build_prediction_citation_dictionary,
    build_run_citation_dictionary,
    freeze_citations,
    merge_citation_dictionaries,
    persisted_context,
    prompt_with_analyzer_context,
)
from .artifacts import (
    artifact_meta,
    list_artifacts,
    read_text_preview,
    resolve_artifact,
    resolve_preview,
)
from .codex_runtime.config import (
    DEFAULT_CODEX_EFFORT,
    DEFAULT_CODEX_FAMILY,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_SERVICE_TIER,
    DEFAULT_SANDBOX,
    SANDBOX_MODES,
    VIBESIM_API_TOKEN,
    codex_family_catalog,
    codex_model,
    codex_model_catalog,
    codex_model_registry,
    normalize_role_runtime,
    prompt_fingerprint,
)
from .codex_runtime.docker import cleanup_conversation
from .codex_runtime.turn import run_turn
from .codex_runtime.workspace import prepare_workspace
from .eval import EvalRequest, run_eval
from .logging_config import compact_text, configure_logging, log_event
from .managed_context import (
    Capability,
    bearer_token,
    capabilities,
    remove_managed_context,
)
from .naming import schedule_auto_naming
from .store import Store
from .turn_result import collect_turn_event, new_turn_result

configure_logging()
LOG = logging.getLogger("vibesim_ui.app")

UI_DIR = Path(__file__).resolve().parents[1]
FRONTEND = UI_DIR / "frontend"
FRONTEND_DIST = FRONTEND / "dist"
SKILL_DOC = UI_DIR / "SKILL.md"

app = FastAPI(title="VibeSim Chat")
store = Store()

_workspace_locks: dict[str, asyncio.Lock] = {}


def _lock_for(workspace_id: str) -> asyncio.Lock:
    """Managed Agent turns are serialized at workspace scope.

    Analyzer reads do not use this lease. Treating every Codex turn as
    potentially mutating is conservative and prevents two conversations from
    racing the same repo before command-level intent is known.
    """
    return _workspace_locks.setdefault(workspace_id, asyncio.Lock())


@dataclass
class ActiveBrowserTurn:
    """A browser turn whose lifetime is independent of any one SSE connection."""

    turn_id: str
    events: list[str] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task[None] | None = None
    finished: bool = False

    async def publish(self, event: str) -> None:
        async with self.condition:
            self.events.append(event)
            self.condition.notify_all()

    async def finish(self) -> None:
        async with self.condition:
            self.finished = True
            self.condition.notify_all()

    async def stream(self) -> AsyncIterator[str]:
        """Replay the turn so a refreshed browser can attach without losing output."""
        cursor = 0
        while True:
            async with self.condition:
                await self.condition.wait_for(
                    lambda: cursor < len(self.events) or self.finished
                )
                pending_events = self.events[cursor:]
                cursor = len(self.events)
                finished = self.finished
            for event in pending_events:
                yield event
            if finished:
                return


_active_browser_turns: dict[tuple[str, str], ActiveBrowserTurn] = {}


def _turn_stream_response(active_turn: ActiveBrowserTurn) -> StreamingResponse:
    return StreamingResponse(
        active_turn.stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def require_token(authorization: str | None = Header(default=None)) -> None:
    """Gate agent-facing endpoints with a bearer token when one is configured.

    No token set -> open (local dev and the same-host eval harness are unaffected).
    Token set -> require `Authorization: Bearer <token>`.
    """
    if not VIBESIM_API_TOKEN:
        return
    if authorization != f"Bearer {VIBESIM_API_TOKEN}":
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_TURN_FAILURES: dict[str, str] = {
    "runtime_storage_full": (
        "The Agent runtime could not start because the host disk is full. "
        "Free space, then retry this question."
    ),
    "upstream_unavailable": (
        "The upstream model service is unavailable, so the Agent could not "
        "produce an answer. This is a service outage, not a problem with your "
        "question. The conversation is intact — continue it to retry."
    ),
    "upstream_rate_limited": (
        "The upstream model service is rate limiting this account, so the Agent "
        "could not produce an answer. The conversation is intact — wait a "
        "moment, then continue it to retry."
    ),
    "codex_call_timeout": (
        "The Agent stalled without producing output and the call was stopped. "
        "The conversation is intact — continue it to retry."
    ),
    "agent_runtime_failure": (
        "The Agent runtime failed before producing an answer. "
        "Retry the question; the full diagnostic is available in the backend log."
    ),
}


def _failure_for_code(code: str) -> dict[str, str]:
    """The stable, user-safe {code, message} contract for one failure code.

    Unknown codes fall back rather than reaching a client verbatim, so a new
    runtime failure can never leak host detail through this path.
    """
    if code not in _TURN_FAILURES:
        code = "agent_runtime_failure"
    return {"code": code, "message": _TURN_FAILURES[code]}


def _turn_failure(exc: Exception) -> dict[str, str]:
    """Map internal runtime exceptions to stable, user-safe failure details.

    The full exception remains in the structured backend log. Browser and agent
    clients receive only this bounded contract so subprocess commands, mounts,
    and other host implementation details never become an apparent answer.
    """

    detail = str(exc).lower()
    if "no space left on device" in detail:
        return _failure_for_code("runtime_storage_full")
    return _failure_for_code("agent_runtime_failure")


_MANAGED_ACTIVITY_KINDS = {
    "simulation.requested",
    "simulation.running",
    "analysis.running",
    "experiment.ready",
    "experiment.failed",
    "experiment.interrupted",
    "job.requested",
    "job.running",
    "job.analysis_running",
    "job.ready",
    "job.failed",
    "job.interrupted",
}


def _managed_turn_activity(workspace_id: str, turn_id: str) -> list[dict]:
    """Project durable managed-run events into the chat timeline contract."""
    activity: list[dict] = []
    for event in store.list_turn_events(
        workspace_id,
        turn_id,
        kinds=_MANAGED_ACTIVITY_KINDS,
    ):
        payload = event["payload"]
        activity.append(_project_job_activity(payload, workspace_id, event["kind"]))
    return activity


def _project_job_activity(
    payload: dict,
    workspace_id: str,
    fallback_status: str,
) -> dict:
    """Preserve the legacy simulation card shape; enrich only typed jobs."""
    activity = {
        "kind": "job",
        "workspaceId": str(payload.get("workspaceId") or workspace_id),
        "status": str(payload.get("status") or fallback_status),
        "experimentId": str(payload.get("experimentId") or ""),
        "experimentPath": str(payload.get("experimentPath") or ""),
        "jobId": str(payload.get("jobId") or ""),
    }
    if payload.get("jobKind"):
        activity.update(
            {
                "jobKind": str(payload["jobKind"]),
                "resourceId": str(payload.get("resourceId") or ""),
                "analyzerResourceId": str(payload.get("analyzerResourceId") or ""),
                "artifactPath": str(payload.get("artifactPath") or ""),
                "descriptor": payload.get("descriptor") or {},
                "summary": payload.get("summary"),
            }
        )
    return activity


def _recoverable_turn_activity(workspace_id: str, turn_id: str) -> list[dict]:
    """Rebuild the durable role timeline from append-only turn events."""
    activity: list[dict] = []
    for event in store.list_turn_events(workspace_id, turn_id):
        kind = event["kind"]
        payload = event["payload"]
        if kind == "intermediate_output":
            activity.append(
                {
                    "kind": kind,
                    "role": str(payload.get("role") or "orchestrator"),
                    "model": str(payload.get("model") or ""),
                    "effort": str(payload.get("effort") or ""),
                    "level": (
                        "milestone"
                        if payload.get("level") == "milestone"
                        else "progress"
                    ),
                    "text": str(payload.get("text") or ""),
                }
            )
        elif kind == "decision":
            activity.append(
                {
                    "kind": kind,
                    "action": str(payload.get("action") or ""),
                    "task": str(payload.get("task") or ""),
                }
            )
        elif kind == "implementer":
            activity.append({"kind": kind, "text": str(payload.get("text") or "")})
        elif kind == "usage":
            activity.append(
                {
                    "kind": kind,
                    "role": str(payload.get("role") or ""),
                    "model": str(payload.get("model") or ""),
                    "effort": str(payload.get("effort") or ""),
                    "duration_ms": int(payload.get("duration_ms") or 0),
                    "tokens": payload.get("tokens") or {},
                }
            )
        elif kind in _MANAGED_ACTIVITY_KINDS:
            activity.append(_project_job_activity(payload, workspace_id, kind))
    return activity


_BACKEND_RESTART_MESSAGE = (
    "This turn was interrupted when the conversation backend restarted. "
    "Completed Agent activity was recovered above, but no final answer was produced. "
    "Send `continue` to resume from the existing workspace and role sessions."
)


def _recover_orphaned_browser_turns() -> int:
    """Finalize pre-restart running rows without inventing an Agent answer."""
    recovered = 0
    for descriptor in store.registry.list(include_archived=True):
        workspace_id = descriptor["workspace_id"]
        for turn in store.list_running_turns(workspace_id):
            activity = _recoverable_turn_activity(workspace_id, turn["id"])
            activity.append({"kind": "error", "text": _BACKEND_RESTART_MESSAGE})
            if store.interrupt_orphaned_turn(
                workspace_id,
                turn["id"],
                turn["conversation_id"],
                content=_BACKEND_RESTART_MESSAGE,
                activity=activity,
            ):
                recovered += 1
                capabilities.revoke_turn(workspace_id, turn["id"])
                remove_managed_context(workspace_id, turn["conversation_id"])
                log_event(
                    LOG,
                    "turn.recovered_after_backend_restart",
                    workspace_id=workspace_id,
                    conversation_id=turn["conversation_id"],
                    turn_id=turn["id"],
                    activity_events=len(activity) - 1,
                )
    return recovered


@app.on_event("startup")
def recover_orphaned_browser_turns_on_startup() -> None:
    _recover_orphaned_browser_turns()


def _latest_turn_citation_dictionary(
    workspace_id: str,
    turn_id: str,
    initial_context: AnalyzerTurnContext | None,
) -> CitationDictionarySnapshot | None:
    """Resolve the immutable dictionary most recently registered by this turn."""
    dynamic_events = store.list_turn_events(
        workspace_id,
        turn_id,
        kinds={"citation.dictionary"},
    )
    if dynamic_events:
        return CitationDictionarySnapshot.model_validate(
            dynamic_events[-1]["payload"]["dictionary"]
        )
    return initial_context.citation_dictionary if initial_context else None


def _autonomous_for_turn(conv: dict, requested_autonomous: bool) -> bool:
    """Lock autonomous mode once the conversation has a user-visible history."""
    if conv.get("messages"):
        return bool(conv.get("autonomous"))
    return requested_autonomous


class RoleRuntime(BaseModel):
    """One role's Codex model, reasoning effort, and speed tier."""

    model: str = DEFAULT_CODEX_MODEL
    effort: str = DEFAULT_CODEX_EFFORT
    service_tier: str = Field(
        default=DEFAULT_CODEX_SERVICE_TIER, alias="serviceTier"
    )

    # `model_` is Pydantic's own namespace; `model` here is a plain field name.
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)


class ConversationRuntime(BaseModel):
    orchestrator: RoleRuntime = Field(default_factory=RoleRuntime)
    implementer: RoleRuntime = Field(default_factory=RoleRuntime)


class UpdateConversationRuntime(BaseModel):
    codex_runtime: ConversationRuntime


class NewConversation(BaseModel):
    sandbox: str = DEFAULT_SANDBOX
    autonomous: bool = False
    codex_runtime: ConversationRuntime = Field(default_factory=ConversationRuntime)
    # Co-evolution (used by vibe-serve): host path to the caller's candidate
    # workspace to bind read-only at /candidate in this conversation's container.
    peer_workspace: str | None = None
    # Materialize the selected workspace repo eagerly at create (instead of
    # lazily on the first message) so a caller can inspect it immediately.
    eager: bool = False


class NewWorkspace(BaseModel):
    display_name: str = Field(alias="displayName")
    auto_name: bool = Field(default=False, alias="autoName")

    model_config = ConfigDict(populate_by_name=True)


class UpdateWorkspace(BaseModel):
    display_name: str | None = Field(default=None, alias="displayName")
    state: str | None = None


class RegisterManagedRun(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    experiment_root: str = Field(alias="experimentRoot")
    run_count: int = Field(default=1, ge=1, alias="runCount")
    axes: list[str] = Field(default_factory=list)


def _role_runtimes(
    selection: ConversationRuntime | dict | None,
) -> dict[str, dict[str, str]]:
    """Coerce a request body or a stored conversation onto the live registry."""
    raw = (
        selection.model_dump()
        if isinstance(selection, ConversationRuntime)
        else (selection or {})
    )
    runtimes: dict[str, dict[str, str]] = {}
    for role in ("orchestrator", "implementer"):
        requested = raw.get(role)
        requested = requested if isinstance(requested, dict) else {}
        if requested.get("model") and requested["model"] not in codex_model_registry():
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "unknown_codex_model",
                    "model": requested["model"],
                },
            )
        runtimes[role] = normalize_role_runtime(
            requested.get("model"),
            requested.get("effort"),
            requested.get("service_tier") or requested.get("serviceTier"),
        )
    return runtimes


def _session_family(event: dict, runtimes: dict[str, dict[str, str]]) -> str:
    """Which auth profile recorded this rollout — the resume-compatibility key."""
    role = str(event.get("role") or "")
    model_id = str(event.get("model") or "") or runtimes.get(role, {}).get("model", "")
    try:
        return codex_model(model_id).family_id
    except ValueError:
        return DEFAULT_CODEX_FAMILY


def _require_available_runtimes(runtimes: dict[str, dict[str, str]]) -> None:
    unavailable = [
        codex_model(selection["model"]).family_id
        for selection in runtimes.values()
        if not codex_model(selection["model"]).available
    ]
    if unavailable:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "codex_family_unavailable",
                "families": sorted(set(unavailable)),
            },
        )


class UpdateManagedRun(BaseModel):
    status: str


class RegisterManagedJob(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_kind: str = Field(alias="jobKind", min_length=1)
    artifact_root: str = Field(alias="artifactRoot", min_length=1)
    analyzer_resource_id: str | None = Field(
        default=None,
        alias="analyzerResourceId",
    )


class UpdateManagedJob(BaseModel):
    status: str


class RegisterManagedCitationDictionary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resource_kind: Literal[
        "aggregate", "run", "prediction", "kernel_profile", "kernel_measurement"
    ] = Field(
        default="aggregate", alias="resourceKind"
    )
    experiment_id: str | None = Field(default=None, alias="experimentId")
    prediction_id: str | None = Field(default=None, alias="predictionId")
    run_id: str | None = Field(default=None, alias="runId")
    profile_id: str | None = Field(default=None, alias="profileId")
    measurement_id: str | None = Field(default=None, alias="measurementId")
    resource_path: str | None = Field(default=None, alias="resourcePath")
    analysis: dict | None = None


class SendMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    sandbox_mode: str = DEFAULT_SANDBOX
    autonomous_mode: bool = False
    analyzer_context: AnalyzerTurnContext | None = Field(
        default=None, alias="analyzerContext"
    )


class AgentSendMessage(BaseModel):
    """One agent turn. `sandbox_mode`/`autonomous_mode` are optional per-turn
    overrides; when omitted they inherit the conversation's create-time settings
    (so a read-only conversation stays read-only unless a turn opts up)."""

    model_config = ConfigDict(populate_by_name=True)

    text: str
    sandbox_mode: str | None = None
    autonomous_mode: bool | None = None
    analyzer_context: AnalyzerTurnContext | None = Field(
        default=None, alias="analyzerContext"
    )


@app.get("/")
def index() -> FileResponse:
    dist_index = FRONTEND_DIST / "index.html"
    if dist_index.exists():
        return FileResponse(dist_index)
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/workspaces")
def list_workspaces() -> dict:
    return {"workspaces": store.registry.list()}


@app.get("/api/codex-backends")
def list_codex_backends() -> dict:
    """Return the safe, server-owned model choices shared by both UIs.

    Models and their effort ladders come from each family's on-disk Codex model
    catalog, so the selector cannot offer something the runtime cannot run.
    """
    default_runtime = normalize_role_runtime(
        DEFAULT_CODEX_MODEL, DEFAULT_CODEX_EFFORT, DEFAULT_CODEX_SERVICE_TIER
    )
    default_runtime = {
        "model": default_runtime["model"],
        "effort": default_runtime["effort"],
        "serviceTier": default_runtime["service_tier"],
    }
    return {
        "models": codex_model_catalog(),
        "families": codex_family_catalog(),
        "defaults": {
            "orchestrator": dict(default_runtime),
            "implementer": dict(default_runtime),
        },
    }


@app.get("/api/conversations")
def list_all_conversations() -> dict:
    """Compatibility index for the standalone shell served at ``/``.

    Every returned row carries its workspace identity. The shell uses that
    identity for all subsequent workspace-scoped operations; this endpoint
    deliberately provides no unscoped write counterpart.
    """

    return {
        "conversations": store.list_all(),
        "sandbox_modes": list(SANDBOX_MODES),
    }


@app.get("/api/jobs")
def list_managed_jobs() -> dict:
    """Return conversation ownership overlays; Analyzer owns result catalogs."""
    fields = {
        "workspace_id",
        "job_id",
        "conversation_id",
        "conversation_title",
        "turn_id",
        "status",
        "job_kind",
        "resource_id",
        "analyzer_resource_id",
        "created_at",
        "updated_at",
    }
    return {
        "jobs": [
            {key: value for key, value in job.items() if key in fields}
            for job in store.list_all_artifact_jobs()
        ]
    }


def _create_workspace(body: NewWorkspace) -> dict:
    descriptor: dict | None = None
    try:
        descriptor = store.registry.create(
            body.display_name,
            naming_state="pending" if body.auto_name else "manual",
        )
        prepare_workspace(descriptor["workspace_id"])
        return descriptor
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        # Workspace creation is one transaction from the client's point of
        # view. Never leave a discoverable descriptor or repo.tmp after a 500.
        if descriptor is not None:
            store.registry.discard_failed_creation(descriptor["workspace_id"])
        raise


@app.post("/api/workspaces")
def create_workspace(body: NewWorkspace) -> dict:
    return _create_workspace(body)


@app.get("/api/workspaces/{workspace_id}")
def get_workspace(workspace_id: str) -> dict:
    try:
        return store.registry.get(workspace_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc


@app.get("/api/workspaces/{workspace_id}/jobs/{resource_id}")
def get_managed_job_resource(workspace_id: str, resource_id: str) -> dict:
    """Return only the conversation-owned overlay for an Analyzer resource.

    Result payloads belong to Analyzer. The browser resolves
    ``analyzerResourceId`` there; this backend never reparses job artifacts.
    """
    job = store.artifact_job_by_resource(workspace_id, resource_id)
    if job is None:
        raise HTTPException(status_code=404, detail="managed job resource not found")
    return {
        "schemaVersion": 1,
        "workspaceId": workspace_id,
        "jobId": job["job_id"],
        "resourceId": job["resource_id"],
        "analyzerResourceId": job["analyzer_resource_id"],
        "jobKind": job["job_kind"],
        "status": job["status"],
    }


_ANALYZER_RESOURCE_PREFIXES = {
    "timing_predict": "p_",
    "kernel_profile": "kp_",
    "kernel_measure": "km_",
}


def _valid_analyzer_resource_id(job_kind: str, resource_id: str | None) -> bool:
    prefix = _ANALYZER_RESOURCE_PREFIXES.get(job_kind)
    if prefix is None or not isinstance(resource_id, str):
        return False
    suffix = resource_id.removeprefix(prefix)
    return (
        resource_id.startswith(prefix)
        and 1 <= len(suffix) <= 64
        and re.fullmatch(r"[a-z0-9_]+", suffix) is not None
    )


@app.patch("/api/workspaces/{workspace_id}")
def update_workspace(workspace_id: str, body: UpdateWorkspace) -> dict:
    try:
        return store.registry.update(
            workspace_id,
            display_name=body.display_name,
            state=body.state,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/workspaces/{workspace_id}/conversations")
def list_conversations(workspace_id: str) -> dict:
    try:
        conversations = store.list(workspace_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc
    return {
        "workspace_id": workspace_id,
        "conversations": conversations,
        "sandbox_modes": list(SANDBOX_MODES),
    }


@app.post("/api/workspaces/{workspace_id}/conversations")
def create_conversation(workspace_id: str, body: NewConversation) -> dict:
    try:
        store.registry.get(workspace_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc
    runtimes = _role_runtimes(body.codex_runtime)
    _require_available_runtimes(runtimes)
    cid = uuid.uuid4().hex[:12]
    fingerprint = prompt_fingerprint(
        autonomous=body.autonomous,
        orchestrator_runtime=runtimes["orchestrator"],
        implementer_runtime=runtimes["implementer"],
    )
    log_event(
        LOG,
        "conversation.create",
        workspace_id=workspace_id,
        conversation_id=cid,
        sandbox=body.sandbox,
        autonomous=body.autonomous,
        codex_runtime=runtimes,
        prompt_fingerprint=fingerprint,
    )
    conversation = store.create(
        workspace_id,
        cid,
        body.sandbox,
        fingerprint,
        autonomous=body.autonomous,
        orchestrator_runtime=runtimes["orchestrator"],
        implementer_runtime=runtimes["implementer"],
        peer_workspace=body.peer_workspace,
        naming_state="pending",
    )
    if body.eager:
        conversation["workspace_path"] = str(prepare_workspace(workspace_id))
    return conversation


@app.patch("/api/workspaces/{workspace_id}/conversations/{cid}/runtime")
def update_conversation_runtime(
    workspace_id: str, cid: str, body: UpdateConversationRuntime
) -> dict:
    runtimes = _role_runtimes(body.codex_runtime)
    _require_available_runtimes(runtimes)
    try:
        store.update_codex_runtime(
            workspace_id,
            cid,
            orchestrator_runtime=runtimes["orchestrator"],
            implementer_runtime=runtimes["implementer"],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="conversation not found") from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "conversation_runtime_locked", "message": str(exc)},
        ) from exc
    conversation = store.get(workspace_id, cid)
    assert conversation is not None
    return conversation


def require_managed_capability(
    authorization: str | None = Header(default=None),
) -> Capability:
    capability = capabilities.authorize(bearer_token(authorization))
    if capability is None:
        raise HTTPException(
            status_code=401,
            detail="missing, expired, or invalid managed-run capability",
        )
    return capability


def _managed_experiment_root(
    capability: Capability,
    requested_root: str,
) -> tuple[Path, str, str]:
    """Resolve a Launcher path into the capability's registered logs root."""
    if not requested_root.strip():
        raise HTTPException(status_code=400, detail="experimentRoot must not be empty")
    repo_root = store.registry.repo_path(capability.workspace_id).resolve()
    logs_root = store.registry.logs_path(capability.workspace_id).resolve()
    requested = Path(requested_root)
    if requested == Path("/workspace") or Path("/workspace") in requested.parents:
        host_path = repo_root / requested.relative_to("/workspace")
        approved_root = requested.as_posix()
    elif requested.is_absolute():
        host_path = requested
        approved_root = requested.as_posix()
    else:
        host_path = repo_root / requested
        approved_root = requested.as_posix()
    resolved = host_path.resolve(strict=False)
    if resolved == logs_root:
        raise HTTPException(
            status_code=400,
            detail="experimentRoot must name a folder below the workspace logs root",
        )
    if logs_root not in resolved.parents:
        raise HTTPException(
            status_code=403,
            detail="experimentRoot is outside the workspace logs root",
        )
    relative_path = resolved.relative_to(logs_root).as_posix()
    return resolved, relative_path, approved_root


def _managed_artifact_root(
    capability: Capability,
    requested_root: str,
) -> tuple[Path, str, str]:
    """Apply the simulation root containment contract to any typed job artifact."""
    try:
        return _managed_experiment_root(capability, requested_root)
    except HTTPException as error:
        detail = str(error.detail).replace("experimentRoot", "artifactRoot")
        raise HTTPException(status_code=error.status_code, detail=detail) from error


@app.post("/api/internal/managed-runs/register")
async def register_managed_run(
    body: RegisterManagedRun,
    capability: Capability = Depends(require_managed_capability),
) -> dict:
    if store.get(capability.workspace_id, capability.conversation_id) is None:
        raise HTTPException(status_code=404, detail="managed conversation not found")
    host_root, relative_path, approved_root = _managed_experiment_root(
        capability,
        body.experiment_root,
    )
    metadata_path = host_root / "experiment.meta.json"
    metadata: dict | None = None
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=409,
                detail="existing experiment metadata could not be read",
            ) from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != 1
            or not isinstance(metadata.get("experiment_id"), str)
            or not metadata["experiment_id"]
        ):
            raise HTTPException(
                status_code=409,
                detail="existing experiment metadata is incompatible",
            )
    database_experiment = store.experiment_by_path(
        capability.workspace_id,
        relative_path,
    )
    metadata_experiment_id = (
        str(metadata["experiment_id"]) if metadata is not None else None
    )
    database_experiment_id = (
        str(database_experiment["id"]) if database_experiment is not None else None
    )
    if (
        metadata_experiment_id is not None
        and database_experiment_id is not None
        and metadata_experiment_id != database_experiment_id
    ):
        raise HTTPException(
            status_code=409,
            detail="filesystem and workspace database disagree on experiment identity",
        )
    experiment_id = (
        metadata_experiment_id or database_experiment_id or f"e_{uuid.uuid4().hex}"
    )
    job = store.create_job(
        capability.workspace_id,
        conversation_id=capability.conversation_id,
        turn_id=capability.turn_id,
        role=capability.role,
        experiment_id=experiment_id,
        experiment_path=relative_path,
    )
    host_root.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        metadata = {
            "schema_version": 1,
            "experiment_id": job["experiment_id"],
            "origin": {
                "kind": "managed",
                "workspace_id": capability.workspace_id,
                "conversation_id": capability.conversation_id,
                "turn_id": capability.turn_id,
                "job_id": job["job_id"],
                "role": capability.role,
            },
        }
        metadata_temporary = metadata_path.with_suffix(".json.tmp")
        metadata_temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            "utf-8",
        )
        metadata_temporary.replace(metadata_path)
    event = {
        "kind": "simulation.requested",
        "workspaceId": capability.workspace_id,
        "conversationId": capability.conversation_id,
        "turnId": capability.turn_id,
        "jobId": job["job_id"],
        "experimentId": job["experiment_id"],
        "experimentPath": relative_path,
        "runCount": body.run_count,
        "axes": body.axes,
    }
    store.append_turn_event(
        capability.workspace_id,
        capability.turn_id,
        "simulation.requested",
        event,
    )
    active_turn = _active_browser_turns.get(
        (capability.workspace_id, capability.conversation_id)
    )
    if active_turn is not None and not active_turn.finished:
        await active_turn.publish(_sse("job", event))
    return {
        "schemaVersion": 1,
        "workspaceId": capability.workspace_id,
        "conversationId": capability.conversation_id,
        "turnId": capability.turn_id,
        "jobId": job["job_id"],
        "experimentId": job["experiment_id"],
        "approvedRoot": approved_root,
    }


@app.post("/api/internal/managed-runs/{job_id}/status")
async def update_managed_run(
    job_id: str,
    body: UpdateManagedRun,
    capability: Capability = Depends(require_managed_capability),
) -> dict:
    allowed_statuses = {
        "running",
        "analysis_running",
        "ready",
        "failed",
        "interrupted",
    }
    if body.status not in allowed_statuses:
        raise HTTPException(status_code=400, detail="unsupported managed-run status")
    job = store.update_job(
        capability.workspace_id,
        job_id,
        status=body.status,
        conversation_id=capability.conversation_id,
        turn_id=capability.turn_id,
    )
    if job is None:
        raise HTTPException(status_code=404, detail="managed job not found")
    event_kind = {
        "running": "simulation.running",
        "analysis_running": "analysis.running",
        "ready": "experiment.ready",
        "failed": "experiment.failed",
        "interrupted": "experiment.interrupted",
    }[body.status]
    event = {
        "kind": event_kind,
        "workspaceId": capability.workspace_id,
        "conversationId": capability.conversation_id,
        "turnId": capability.turn_id,
        "jobId": job_id,
        "experimentId": job["experiment_id"],
        "experimentPath": job["experiment_path"],
        "status": body.status,
    }
    store.append_turn_event(
        capability.workspace_id,
        capability.turn_id,
        event_kind,
        event,
    )
    active_turn = _active_browser_turns.get(
        (capability.workspace_id, capability.conversation_id)
    )
    if active_turn is not None and not active_turn.finished:
        await active_turn.publish(_sse("job", event))
    return event


@app.post("/api/internal/managed-jobs/register")
async def register_managed_job(
    body: RegisterManagedJob,
    capability: Capability = Depends(require_managed_capability),
) -> dict:
    allowed_job_kinds = {"timing_predict", "kernel_profile", "kernel_measure"}
    if body.job_kind not in allowed_job_kinds:
        raise HTTPException(status_code=400, detail="unsupported managed job kind")
    if not _valid_analyzer_resource_id(body.job_kind, body.analyzer_resource_id):
        raise HTTPException(
            status_code=400,
            detail=f"{body.job_kind} requires a valid analyzerResourceId",
        )
    if store.get(capability.workspace_id, capability.conversation_id) is None:
        raise HTTPException(status_code=404, detail="managed conversation not found")
    _host_root, relative_path, approved_root = _managed_artifact_root(
        capability,
        body.artifact_root,
    )
    job = store.create_artifact_job(
        capability.workspace_id,
        conversation_id=capability.conversation_id,
        turn_id=capability.turn_id,
        role=capability.role,
        job_kind=body.job_kind,
        artifact_path=relative_path,
        analyzer_resource_id=body.analyzer_resource_id,
    )
    event = {
        "kind": "job.requested",
        "workspaceId": capability.workspace_id,
        "conversationId": capability.conversation_id,
        "turnId": capability.turn_id,
        "jobId": job["job_id"],
        "jobKind": job["job_kind"],
        "resourceId": job["resource_id"],
        "analyzerResourceId": job["analyzer_resource_id"],
        "status": "requested",
    }
    store.append_turn_event(
        capability.workspace_id,
        capability.turn_id,
        "job.requested",
        event,
    )
    active_turn = _active_browser_turns.get(
        (capability.workspace_id, capability.conversation_id)
    )
    if active_turn is not None and not active_turn.finished:
        await active_turn.publish(_sse("job", event))
    return {
        "schemaVersion": 1,
        "workspaceId": capability.workspace_id,
        "conversationId": capability.conversation_id,
        "turnId": capability.turn_id,
        "jobId": job["job_id"],
        "resourceId": job["resource_id"],
        "analyzerResourceId": job["analyzer_resource_id"],
        "approvedRoot": approved_root,
    }


@app.post("/api/internal/managed-jobs/{job_id}/status")
async def update_managed_job(
    job_id: str,
    body: UpdateManagedJob,
    capability: Capability = Depends(require_managed_capability),
) -> dict:
    allowed_statuses = {
        "running",
        "analysis_running",
        "ready",
        "failed",
        "interrupted",
    }
    if body.status not in allowed_statuses:
        raise HTTPException(status_code=400, detail="unsupported managed-job status")
    job = store.update_job(
        capability.workspace_id,
        job_id,
        status=body.status,
        conversation_id=capability.conversation_id,
        turn_id=capability.turn_id,
    )
    if job is None or job["job_kind"] == "simulation":
        raise HTTPException(status_code=404, detail="managed job not found")
    event_kind = f"job.{body.status}"
    event = {
        "kind": event_kind,
        "workspaceId": capability.workspace_id,
        "conversationId": capability.conversation_id,
        "turnId": capability.turn_id,
        "jobId": job_id,
        "jobKind": job["job_kind"],
        "resourceId": job["resource_id"],
        "analyzerResourceId": job["analyzer_resource_id"],
        "status": body.status,
    }
    store.append_turn_event(
        capability.workspace_id,
        capability.turn_id,
        event_kind,
        event,
    )
    active_turn = _active_browser_turns.get(
        (capability.workspace_id, capability.conversation_id)
    )
    if active_turn is not None and not active_turn.finished:
        await active_turn.publish(_sse("job", event))
    return event


@app.post("/api/internal/analyzer-citations/register")
def register_managed_analyzer_citations(
    body: RegisterManagedCitationDictionary,
    capability: Capability = Depends(require_managed_capability),
) -> dict:
    """Register evidence discovered after an Agent-first turn has started."""
    try:
        if body.resource_kind == "run":
            if body.run_id is None or body.resource_path is None:
                raise ValueError("run citation registration requires runId and resourcePath")
            dictionary = build_run_citation_dictionary(
                workspace_id=capability.workspace_id,
                run_id=body.run_id,
                resource_path=body.resource_path,
            )
            resource_id = body.run_id
        elif body.resource_kind == "prediction":
            if body.prediction_id is None or body.resource_path is None:
                raise ValueError(
                    "prediction citation registration requires predictionId and resourcePath"
                )
            dictionary = build_prediction_citation_dictionary(
                workspace_id=capability.workspace_id,
                prediction_id=body.prediction_id,
                resource_path=body.resource_path,
            )
            resource_id = body.prediction_id
        elif body.resource_kind == "kernel_profile":
            if body.profile_id is None or body.resource_path is None or body.analysis is None:
                raise ValueError(
                    "kernel profile citation registration requires profileId, "
                    "resourcePath, and analysis"
                )
            dictionary = build_kernel_profile_citation_dictionary(
                workspace_id=capability.workspace_id,
                profile_id=body.profile_id,
                resource_path=body.resource_path,
                analysis=body.analysis,
            )
            resource_id = body.profile_id
        elif body.resource_kind == "kernel_measurement":
            if (
                body.measurement_id is None
                or body.resource_path is None
                or body.analysis is None
            ):
                raise ValueError(
                    "kernel measurement citation registration requires measurementId, "
                    "resourcePath, and analysis"
                )
            dictionary = build_kernel_measurement_citation_dictionary(
                workspace_id=capability.workspace_id,
                measurement_id=body.measurement_id,
                resource_path=body.resource_path,
                analysis=body.analysis,
            )
            resource_id = body.measurement_id
        else:
            if body.experiment_id is None or body.analysis is None:
                raise ValueError(
                    "aggregate citation registration requires experimentId and analysis"
                )
            workspace_experiment = store.get_experiment(
                capability.workspace_id,
                body.experiment_id,
            )
            if workspace_experiment is not None and workspace_experiment["status"] != "ready":
                raise HTTPException(
                    status_code=409,
                    detail="Analyzer experiment is not ready",
                )
            # Analyzer catalogs are shared read-only evidence. Conversation or
            # workspace ownership controls mutation, not whether a ready result
            # may be cited from another managed turn.
            dictionary = build_aggregate_citation_dictionary(
                body.analysis,
                workspace_id=capability.workspace_id,
                experiment_id=body.experiment_id,
            )
            resource_id = body.experiment_id
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dictionary = merge_citation_dictionaries(
        _latest_turn_citation_dictionary(
            capability.workspace_id,
            capability.turn_id,
            None,
        ),
        dictionary,
    )
    # EvidenceRef nullable selectors are required protocol fields. Keep their
    # explicit nulls in the response/event so the final-answer freezer can
    # validate the persisted dictionary again after the MCP call returns.
    snapshot = dictionary.model_dump(by_alias=True)
    store.append_turn_event(
        capability.workspace_id,
        capability.turn_id,
        "citation.dictionary",
        {
            "resourceKind": body.resource_kind,
            "resourceId": resource_id,
            **{
                "aggregate": {"experimentId": resource_id},
                "run": {"runId": resource_id},
                "prediction": {"predictionId": resource_id},
                "kernel_profile": {"profileId": resource_id},
                "kernel_measurement": {"measurementId": resource_id},
            }[body.resource_kind],
            "dictionary": snapshot,
        },
    )
    return snapshot


@app.get("/api/workspaces/{workspace_id}/conversations/{cid}/experiments")
def list_conversation_experiments(workspace_id: str, cid: str) -> dict:
    if store.get(workspace_id, cid) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {
        "workspace_id": workspace_id,
        "conversation_id": cid,
        "experiments": store.list_experiments(
            workspace_id,
            conversation_id=cid,
        ),
    }


@app.post("/api/eval")
async def eval_run(body: EvalRequest, _: None = Depends(require_token)) -> dict:
    try:
        return await run_eval(
            prompt=body.prompt,
            sandbox=body.sandbox,
            autonomous=body.autonomous,
            keep_container=body.keep_container,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/agent/skill")
def serve_skill() -> FileResponse:
    """Public agent-facing skill doc. Fetch this first to learn what VibeSim does,
    when to call it, what to expect, and the HTTP contract."""
    if not SKILL_DOC.is_file():
        raise HTTPException(status_code=404, detail="SKILL.md not found")
    return FileResponse(str(SKILL_DOC), media_type="text/markdown")


@app.get("/api/agent/workspaces")
def agent_list_workspaces(_: None = Depends(require_token)) -> dict:
    return {"workspaces": store.registry.list()}


@app.post("/api/agent/workspaces")
def agent_create_workspace(
    body: NewWorkspace,
    _: None = Depends(require_token),
) -> dict:
    return _create_workspace(body)


# FileNotFoundError/PermissionError/ValueError raised by artifacts.py map to
# 404/403/400 so the agent gets an actionable status without HTTP leaking into
# that module.
def _artifact_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/api/agent/workspaces/{workspace_id}/artifacts")
def list_workspace_artifacts(
    workspace_id: str,
    subdir: str | None = Query(default=None),
    limit: int = Query(default=2000, ge=1, le=20000),
    _: None = Depends(require_token),
) -> dict:
    try:
        return list_artifacts(workspace_id, subdir=subdir, limit=limit)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise _artifact_http_error(exc) from exc


@app.get("/api/agent/workspaces/{workspace_id}/artifacts/download")
def download_workspace_artifact(
    workspace_id: str,
    path: str = Query(...),
    _: None = Depends(require_token),
) -> FileResponse:
    try:
        resolved = resolve_artifact(workspace_id, path)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise _artifact_http_error(exc) from exc
    return FileResponse(str(resolved), filename=resolved.name)


# --- Agent conversation API: the real interactive interface -----------------
# Multi-turn, synchronous JSON, token-gated. Reuses the same `store` and
# `run_turn` as the browser SSE path, so a conversation keeps its isolated
# workspace and resumes its orchestrator/implementer Codex sessions across turns.
# The calling agent reads `final` and, like a human, decides whether to answer a
# clarifying question or steer with another turn. Artifacts are fetched through
# the workspace-scoped /api/agent/workspaces/{workspace_id}/artifacts* routes.


@app.post("/api/agent/workspaces/{workspace_id}/conversations")
def agent_create_conversation(
    workspace_id: str,
    body: NewConversation,
    _: None = Depends(require_token),
) -> dict:
    try:
        store.registry.get(workspace_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc
    runtimes = _role_runtimes(body.codex_runtime)
    _require_available_runtimes(runtimes)
    cid = uuid.uuid4().hex[:12]
    fingerprint = prompt_fingerprint(
        autonomous=body.autonomous,
        orchestrator_runtime=runtimes["orchestrator"],
        implementer_runtime=runtimes["implementer"],
    )
    log_event(
        LOG,
        "agent.conversation.create",
        workspace_id=workspace_id,
        conversation_id=cid,
        sandbox=body.sandbox,
        autonomous=body.autonomous,
        codex_runtime=runtimes,
        prompt_fingerprint=fingerprint,
        peer_workspace=body.peer_workspace,
        eager=body.eager,
    )
    conv = store.create(
        workspace_id,
        cid,
        body.sandbox,
        fingerprint,
        autonomous=body.autonomous,
        orchestrator_runtime=runtimes["orchestrator"],
        implementer_runtime=runtimes["implementer"],
        peer_workspace=body.peer_workspace,
        naming_state="pending",
    )
    if body.eager:
        # Materialize the isolated workspace now so the caller can bind-mount it
        # read-only before its own container launches (the container itself is
        # still started lazily on the first message). Return its host path.
        workspace_repo = prepare_workspace(workspace_id)
        conv["workspace_path"] = str(workspace_repo)
    return conv


@app.get("/api/agent/workspaces/{workspace_id}/conversations/{cid}")
def agent_get_conversation(
    workspace_id: str, cid: str, _: None = Depends(require_token)
) -> dict:
    conv = store.get(workspace_id, cid)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@app.delete("/api/agent/workspaces/{workspace_id}/conversations/{cid}")
def agent_delete_conversation(
    workspace_id: str, cid: str, _: None = Depends(require_token)
) -> dict:
    log_event(
        LOG,
        "agent.conversation.delete",
        workspace_id=workspace_id,
        conversation_id=cid,
    )
    store.delete(workspace_id, cid)
    cleanup_conversation(workspace_id, cid)
    return {"ok": True}


@app.post("/api/agent/workspaces/{workspace_id}/conversations/{cid}/messages")
async def agent_send_message(
    workspace_id: str,
    cid: str,
    body: AgentSendMessage,
    _: None = Depends(require_token),
) -> dict:
    conv = store.get(workspace_id, cid)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty message")

    # Per-turn overrides fall back to the conversation's create-time settings.
    sandbox = body.sandbox_mode or conv.get("sandbox") or DEFAULT_SANDBOX
    requested_autonomous = (
        body.autonomous_mode
        if body.autonomous_mode is not None
        else bool(conv.get("autonomous", False))
    )
    autonomous = _autonomous_for_turn(conv, requested_autonomous)
    runtimes = _role_runtimes(conv.get("codex_runtime") or {})
    fingerprint = prompt_fingerprint(
        autonomous=autonomous,
        orchestrator_runtime=runtimes["orchestrator"],
        implementer_runtime=runtimes["implementer"],
    )
    turn_id = uuid.uuid4().hex[:10]
    store.update_runtime_settings(
        workspace_id,
        cid,
        sandbox=sandbox,
        autonomous=autonomous,
    )
    store.add_message(
        workspace_id,
        cid,
        "user",
        text,
        analyzer_context=persisted_context(body.analyzer_context),
    )
    sessions = store.sessions_for_prompt(
        workspace_id, cid, fingerprint, runtimes=runtimes
    )
    store.start_turn(workspace_id, cid, turn_id)
    log_event(
        LOG,
        "agent.turn.received",
        workspace_id=workspace_id,
        conversation_id=cid,
        turn_id=turn_id,
        sandbox=sandbox,
        autonomous=autonomous,
        prompt_fingerprint=fingerprint,
        prompt_len=len(text),
        prompt_preview=compact_text(text),
        active_session_roles=sorted(sessions),
    )

    result = new_turn_result(
        conversation_id=cid,
        turn_id=turn_id,
        sandbox=sandbox,
        autonomous=autonomous,
    )

    lock = _lock_for(workspace_id)
    async with lock:
        try:
            async for ev in run_turn(
                workspace_id,
                cid,
                prompt_with_analyzer_context(text, body.analyzer_context),
                sandbox=sandbox,
                sessions=sessions,
                turn_id=turn_id,
                prompt_fingerprint=fingerprint,
                autonomous=autonomous,
                peer_dir=conv.get("peer_workspace"),
                orchestrator_runtime=runtimes["orchestrator"],
                implementer_runtime=runtimes["implementer"],
            ):
                if ev.get("kind") == "session":
                    store.set_codex_session(
                        workspace_id,
                        cid,
                        ev.get("role", ""),
                        ev.get("session_id"),
                        family=_session_family(ev, runtimes),
                    )
                store.append_turn_event(
                    workspace_id,
                    turn_id,
                    str(ev.get("kind") or "event"),
                    ev,
                )
                collect_turn_event(result, ev)
            result["ok"] = (
                bool(result["final"])
                and not bool(result["error"])
                and not bool(result["failure_code"])
            )
        except Exception as exc:  # surface backend failures in-band, like /api/eval
            result["error"] = str(exc)
            LOG.exception(
                "agent.turn.error",
                extra={
                    "event_fields": {
                        "event": "agent.turn.error",
                        "conversation_id": cid,
                        "turn_id": turn_id,
                        "error": str(exc),
                    }
                },
            )
        if result["failure_code"]:
            # A runtime-classified failure (upstream outage, idle timeout): the
            # browser path publishes the same contract for the same event.
            failure = _failure_for_code(result["failure_code"])
        elif result["error"]:
            failure = _turn_failure(RuntimeError(result["error"]))
        else:
            failure = None
        result["failure"] = failure
        if not result["final"]:
            result["final"] = failure["message"] if failure else "(no answer)"
        if failure is None and result["outcome"] is None:
            result["outcome"] = "final_answer"
        citation_dictionary = _latest_turn_citation_dictionary(
            workspace_id, turn_id, body.analyzer_context
        )
        frozen_citations = freeze_citations(result["final"], citation_dictionary)
        result["citations"] = frozen_citations
        result["citation_dictionary_id"] = (
            citation_dictionary.identity if citation_dictionary else None
        )
        activity = _managed_turn_activity(workspace_id, turn_id)
        activity.append(
            {
                "kind": "error" if failure else "final",
                "text": result["final"],
                **(
                    {"outcome": result["outcome"]}
                    if failure is None and result["outcome"] is not None
                    else {}
                ),
            }
        )
        store.add_message(
            workspace_id,
            cid,
            "assistant",
            result["final"],
            intermediate_outputs=result["intermediate_outputs"] or None,
            activity=activity,
            citations=frozen_citations or None,
            citation_dictionary_id=result["citation_dictionary_id"],
            citation_dsl_version="v2" if citation_dictionary else None,
            failure=failure,
        )
        store.finish_turn(
            workspace_id,
            turn_id,
            "complete" if result["ok"] else "failed",
        )
        result["naming_scheduled"] = (
            schedule_auto_naming(
                store,
                workspace_id,
                cid,
                text,
                result["final"],
            )
            if result["ok"]
            else False
        )
        capabilities.revoke_turn(workspace_id, turn_id)
        remove_managed_context(workspace_id, cid)
    log_event(
        LOG,
        "agent.turn.complete",
        workspace_id=workspace_id,
        conversation_id=cid,
        turn_id=turn_id,
        ok=result["ok"],
        final_len=len(result["final"]),
        final_preview=compact_text(result["final"]),
    )
    return result


@app.get("/api/workspaces/{workspace_id}/conversations/{cid}")
def get_conversation(
    workspace_id: str,
    cid: str,
    limit: int | None = Query(default=None, ge=1, le=100),
    before: int | None = Query(default=None, ge=0),
) -> dict:
    """Return a full conversation or a backwards page for the browser UI.

    The parameterless response remains the original full-history contract.
    Supplying ``limit`` opts into pagination; ``before`` is the exclusive
    absolute message position returned as ``message_page.start_index`` by the
    newer page. The store remains append-only and is never trimmed by reads.
    """
    if limit is None and before is not None:
        raise HTTPException(status_code=422, detail="before requires limit")
    conv = (
        store.get_message_page(
            workspace_id,
            cid,
            before=before,
            limit=limit,
        )
        if limit is not None
        else store.get(workspace_id, cid)
    )
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@app.delete("/api/workspaces/{workspace_id}/conversations/{cid}")
def delete_conversation(workspace_id: str, cid: str) -> dict:
    log_event(
        LOG,
        "conversation.delete",
        workspace_id=workspace_id,
        conversation_id=cid,
    )
    store.delete(workspace_id, cid)
    cleanup_conversation(workspace_id, cid)
    return {"ok": True}


# Serve files the assistant references in markdown or in its prose (generated
# plots, run summaries, source it just edited). Relative paths resolve against
# the workspace repo (the assistant's workdir); `/workspace/...` container paths
# are accepted verbatim. These three routes are NOT token-gated, so they go
# through `guard_preview`: workspace containment plus a build/VCS/credential
# denylist. Reading is always bounded — see artifacts.MAX_PREVIEW_BYTES.


@app.get("/api/file")
def serve_file(
    path: str = Query(...), workspace_id: str = Query(default="w_main")
) -> Response:
    try:
        resolved, kind = resolve_preview(workspace_id, path)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise _artifact_http_error(exc) from exc
    if kind == "image":
        return FileResponse(str(resolved))
    if kind == "binary":
        return FileResponse(
            str(resolved), filename=resolved.name, media_type="application/octet-stream"
        )
    text, truncated = read_text_preview(resolved)
    return Response(
        content=text,
        media_type="text/plain; charset=utf-8",
        headers={
            "X-File-Truncated": "1" if truncated else "0",
            "X-File-Total-Bytes": str(resolved.stat().st_size),
        },
    )


@app.get("/api/file/meta")
def serve_file_meta(
    path: str = Query(...), workspace_id: str = Query(default="w_main")
) -> dict:
    try:
        return artifact_meta(workspace_id, path)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise _artifact_http_error(exc) from exc


@app.get("/api/file/list")
def serve_file_list(
    path: str | None = Query(default=None),
    workspace_id: str = Query(default="w_main"),
    limit: int = Query(default=2000, ge=1, le=20000),
) -> dict:
    try:
        return list_artifacts(
            workspace_id, subdir=path, limit=limit, recursive=False, preview=True
        )
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise _artifact_http_error(exc) from exc


@app.post("/api/workspaces/{workspace_id}/conversations/{cid}/messages")
async def send_message(
    workspace_id: str, cid: str, body: SendMessage
) -> StreamingResponse:
    conv = store.get(workspace_id, cid)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    active_key = (workspace_id, cid)
    active_turn = _active_browser_turns.get(active_key)
    if active_turn is not None and not active_turn.finished:
        raise HTTPException(
            status_code=409, detail="conversation already has an active turn"
        )

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty message")

    sandbox = body.sandbox_mode
    autonomous = _autonomous_for_turn(conv, body.autonomous_mode)
    runtimes = _role_runtimes(conv.get("codex_runtime") or {})
    current_prompt_fingerprint = prompt_fingerprint(
        autonomous=autonomous,
        orchestrator_runtime=runtimes["orchestrator"],
        implementer_runtime=runtimes["implementer"],
    )
    turn_id = uuid.uuid4().hex[:10]
    previous_sessions = dict(conv.get("codex_sessions") or {})
    store.update_runtime_settings(
        workspace_id,
        cid,
        sandbox=sandbox,
        autonomous=autonomous,
    )
    store.add_message(
        workspace_id,
        cid,
        "user",
        text,
        analyzer_context=persisted_context(body.analyzer_context),
    )
    sessions = store.sessions_for_prompt(
        workspace_id,
        cid,
        current_prompt_fingerprint,
        runtimes=runtimes,
    )
    store.start_turn(workspace_id, cid, turn_id)
    log_event(
        LOG,
        "turn.received",
        workspace_id=workspace_id,
        conversation_id=cid,
        turn_id=turn_id,
        sandbox=sandbox,
        autonomous=autonomous,
        prompt_fingerprint=current_prompt_fingerprint,
        prompt_len=len(text),
        prompt_preview=compact_text(text),
        previous_session_roles=sorted(previous_sessions),
        active_session_roles=sorted(sessions),
        sessions_reset=bool(previous_sessions) and not sessions,
    )

    active_turn = ActiveBrowserTurn(turn_id=turn_id)
    _active_browser_turns[active_key] = active_turn
    active_turn.task = asyncio.create_task(
        _run_browser_turn(
            cid=cid,
            workspace_id=workspace_id,
            text=text,
            sandbox=sandbox,
            sessions=sessions,
            turn_id=turn_id,
            prompt_fingerprint=current_prompt_fingerprint,
            autonomous=autonomous,
            analyzer_context=body.analyzer_context,
            active_turn=active_turn,
            runtimes=runtimes,
        )
    )
    return _turn_stream_response(active_turn)


@app.get("/api/workspaces/{workspace_id}/conversations/{cid}/stream")
async def resume_message_stream(workspace_id: str, cid: str) -> Response:
    if store.get(workspace_id, cid) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    active_turn = _active_browser_turns.get((workspace_id, cid))
    if active_turn is None or active_turn.finished:
        # Reopening an idle conversation is a normal state, not a request
        # conflict. A body-less response also avoids a noisy console error.
        return Response(status_code=204)
    return _turn_stream_response(active_turn)


@app.post("/api/workspaces/{workspace_id}/conversations/{cid}/cancel")
async def cancel_message(workspace_id: str, cid: str) -> dict:
    if store.get(workspace_id, cid) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    active_turn = _active_browser_turns.get((workspace_id, cid))
    if active_turn is None or active_turn.task is None or active_turn.task.done():
        return {"cancelled": False}
    active_turn.task.cancel()
    try:
        await active_turn.task
    except asyncio.CancelledError:
        pass
    return {"cancelled": True}


async def _run_browser_turn(
    *,
    workspace_id: str,
    cid: str,
    text: str,
    sandbox: str,
    sessions: dict[str, str],
    turn_id: str,
    prompt_fingerprint: str,
    autonomous: bool,
    analyzer_context: AnalyzerTurnContext | None,
    active_turn: ActiveBrowserTurn,
    runtimes: dict[str, dict[str, str]] | None = None,
) -> None:
    runtimes = runtimes or _role_runtimes(None)
    lock = _lock_for(workspace_id)
    async with lock:
        final_text: str | None = None
        final_outcome: str | None = None
        failure: dict[str, str] | None = None
        cancelled = False
        intermediate_outputs: list[dict[str, str]] = []
        # Persist render-relevant events on completion; retain all SSE events in
        # ActiveBrowserTurn during execution so a refreshed client can replay them.
        activity: list[dict] = []
        try:
            async for ev in run_turn(
                workspace_id,
                cid,
                prompt_with_analyzer_context(text, analyzer_context),
                sandbox=sandbox,
                sessions=sessions,
                turn_id=turn_id,
                prompt_fingerprint=prompt_fingerprint,
                autonomous=autonomous,
                orchestrator_runtime=runtimes["orchestrator"],
                implementer_runtime=runtimes["implementer"],
            ):
                kind = ev.get("kind")
                if kind == "session":
                    store.set_codex_session(
                        workspace_id,
                        cid,
                        ev.get("role", ""),
                        ev.get("session_id"),
                        family=_session_family(ev, runtimes),
                    )
                    log_event(
                        LOG,
                        "turn.session",
                        conversation_id=cid,
                        turn_id=turn_id,
                        role=ev.get("role", ""),
                        codex_session_id=ev.get("session_id"),
                    )
                    await active_turn.publish(
                        _sse(
                            "session",
                            {
                                "role": ev.get("role"),
                                "backend": ev.get("backend"),
                                "session_id": ev.get("session_id"),
                            },
                        )
                    )
                elif kind == "implementer":
                    implementer_text = str(ev.get("text") or "")
                    activity.append({"kind": "implementer", "text": implementer_text})
                    await active_turn.publish(
                        _sse("implementer", {"text": implementer_text})
                    )
                elif kind == "intermediate_output":
                    intermediate_output = {
                        "role": str(ev.get("role") or ""),
                        "model": str(ev.get("model") or ""),
                        "effort": str(ev.get("effort") or ""),
                        "level": (
                            "milestone"
                            if ev.get("level") == "milestone"
                            else "progress"
                        ),
                        "text": str(ev.get("text") or ""),
                    }
                    intermediate_outputs.append(intermediate_output)
                    activity.append(
                        {"kind": "intermediate_output", **intermediate_output}
                    )
                    await active_turn.publish(
                        _sse("intermediate_output", intermediate_output)
                    )
                elif kind == "decision":
                    decision = {
                        "action": str(ev.get("action") or ""),
                        "task": str(ev.get("task") or ""),
                    }
                    activity.append({"kind": "decision", **decision})
                    await active_turn.publish(_sse("decision", decision))
                elif kind == "usage":
                    usage = {
                        "role": str(ev.get("role") or ""),
                        "model": str(ev.get("model") or ""),
                        "effort": str(ev.get("effort") or ""),
                        "duration_ms": int(ev.get("duration_ms") or 0),
                        "tokens": ev.get("tokens") or {},
                    }
                    activity.append({"kind": "usage", **usage})
                    await active_turn.publish(_sse("usage", usage))
                elif kind == "tool_call":
                    await active_turn.publish(
                        _sse("tool_call", {"text": ev.get("text", "")})
                    )
                elif kind == "error":
                    await active_turn.publish(
                        _sse("error", {"text": ev.get("text", "")})
                    )
                elif kind == "final":
                    final_text = ev.get("text") or ""
                    event_failure = ev.get("failure")
                    if isinstance(event_failure, dict):
                        # No orchestrator decision was ever made, so there is no
                        # outcome to report. `failure` is what the rest of this
                        # function already keys the error card, the `failed`
                        # status and the suppressed auto-naming off. The runtime's
                        # own text is kept as the body: it names the status and
                        # the surviving work, and is already free of host detail.
                        failure = _failure_for_code(str(event_failure.get("code") or ""))
                        final_outcome = None
                    else:
                        outcome = ev.get("outcome")
                        final_outcome = (
                            outcome
                            if outcome in {"final_answer", "request_user_input"}
                            else "final_answer"
                        )
                store.append_turn_event(
                    workspace_id,
                    turn_id,
                    str(kind or "event"),
                    ev,
                )
            if final_text is None:
                final_text = "(no answer)"
                final_outcome = "final_answer"
            log_event(
                LOG,
                "turn.complete",
                conversation_id=cid,
                turn_id=turn_id,
                final_len=len(final_text),
                final_preview=compact_text(final_text),
                failure_code=failure["code"] if failure else "",
            )
        except asyncio.CancelledError:
            cancelled = True
            final_text = "Stopped."
            final_outcome = None
            log_event(
                LOG,
                "turn.cancelled",
                conversation_id=cid,
                turn_id=turn_id,
            )
        except Exception as exc:  # surface backend failures to the UI
            failure = _turn_failure(exc)
            final_text = failure["message"]
            final_outcome = None
            LOG.exception(
                "turn.error",
                extra={
                    "event_fields": {
                        "event": "turn.error",
                        "conversation_id": cid,
                        "turn_id": turn_id,
                        "error": str(exc),
                    }
                },
            )
        finally:
            if final_text is not None:
                citation_dictionary = _latest_turn_citation_dictionary(
                    workspace_id, turn_id, analyzer_context
                )
                frozen_citations = freeze_citations(final_text, citation_dictionary)
                citation_dictionary_id = (
                    citation_dictionary.identity if citation_dictionary else None
                )
                activity.append(
                    {
                        "kind": "error" if failure else "final",
                        "text": final_text,
                        **(
                            {"outcome": final_outcome}
                            if not failure and final_outcome is not None
                            else {}
                        ),
                    }
                )
                managed_activity = _managed_turn_activity(workspace_id, turn_id)
                if managed_activity:
                    activity[-1:-1] = managed_activity
                store.add_message(
                    workspace_id,
                    cid,
                    "assistant",
                    final_text,
                    intermediate_outputs=intermediate_outputs or None,
                    activity=activity or None,
                    citations=frozen_citations or None,
                    citation_dictionary_id=citation_dictionary_id,
                    citation_dsl_version="v2" if citation_dictionary else None,
                    failure=failure,
                )
                naming_scheduled = (
                    schedule_auto_naming(
                        store,
                        workspace_id,
                        cid,
                        text,
                        final_text,
                    )
                    if failure is None and not cancelled and final_text != "(no answer)"
                    else False
                )
                await active_turn.publish(
                    _sse(
                        "done",
                        {
                            "text": final_text,
                            "outcome": final_outcome,
                            "citations": frozen_citations,
                            "citation_dictionary_id": citation_dictionary_id,
                            "citation_dsl_version": "v2" if analyzer_context else None,
                            "failure": failure,
                            "naming_scheduled": naming_scheduled,
                        },
                    )
                )
            await active_turn.finish()
            store.finish_turn(
                workspace_id,
                turn_id,
                "failed" if failure else "complete",
            )
            capabilities.revoke_turn(workspace_id, turn_id)
            remove_managed_context(workspace_id, cid)
            active_key = (workspace_id, cid)
            if _active_browser_turns.get(active_key) is active_turn:
                _active_browser_turns.pop(active_key, None)


# Mounted last so it doesn't shadow the API routes above. The directory may not
# exist before the first frontend build; `run.sh` creates it for normal serving.
app.mount(
    "/assets",
    StaticFiles(directory=str(FRONTEND_DIST / "assets"), check_dir=False),
    name="assets",
)
