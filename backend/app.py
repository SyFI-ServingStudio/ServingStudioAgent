"""FastAPI app: REST + SSE chat over Docker-backed `codex exec`.

Endpoints:
  GET    /                              -> Vite frontend index
  GET    /assets/*                      -> Vite frontend assets
  GET    /api/conversations             -> list (id, title, updated_at)
  POST   /api/conversations             -> create empty conversation
  GET    /api/conversations/{cid}       -> full or cursor-paged conversation
  DELETE /api/conversations/{cid}       -> delete
  POST   /api/conversations/{cid}/messages  -> SSE stream of one turn
  GET    /api/conversations/{cid}/stream    -> reconnect to the active turn
  POST   /api/conversations/{cid}/cancel    -> cancel the active turn
  POST   /api/eval                          -> JSON single-turn eval (evaluation only)
  GET    /api/agent/skill                    -> agent skill doc (SKILL.md, public)
  GET    /api/agent/artifacts                -> list files in a run's workspace
  GET    /api/agent/artifacts/download       -> download one workspace file
  POST   /api/agent/conversations            -> create an agent conversation
  GET    /api/agent/conversations/{cid}      -> full agent conversation history
  DELETE /api/agent/conversations/{cid}      -> delete an agent conversation
  POST   /api/agent/conversations/{cid}/messages -> run one turn, synchronous JSON

All agent-facing endpoints share the /api/agent/* prefix and are gated by
`require_token` when VIBESIM_API_TOKEN is set; /api/agent/skill stays public so an
agent can learn the contract before it holds a token. The
`/api/agent/conversations*` endpoints are the real interactive interface
(multi-turn, session + workspace continuity, reusing `store` and `run_turn`);
`/api/eval` is single-turn and evaluation-only (kept outside the /api/agent/*
namespace on purpose). The browser SSE endpoints (/api/conversations*) are
unchanged and not token-gated.

Browser turns run independently of any one HTTP connection, so a refreshed page
can replay and continue the active SSE stream. The message endpoint streams
Server-Sent Events: `session` (role Codex session id),
`progress` (transient activity lines), `intermediate_output` (assistant commentary),
`decision` (orchestrator→implementer delegated task), `usage` (per-call duration +
token breakdown), `implementer` (implementer summary), then `done` (stored final
answer). `implementer`/`decision`/`usage` can repeat inside one browser turn when the
orchestrator issues follow-up tasks. The render-relevant events are also persisted as
an ordered `activity` list on the assistant message so a reload rebuilds the same role
timeline. A per-conversation lock prevents two turns racing the same copied
workspace/container.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .artifacts import list_artifacts, resolve_artifact
from .codex_runtime.config import (
    DEFAULT_SANDBOX,
    MAIN_DIR,
    SANDBOX_MODES,
    VIBESIM_API_TOKEN,
    WORKSPACE,
    WORKSPACES_DIR,
    prompt_fingerprint,
    workspace_main_for,
)
from .codex_runtime.docker import cleanup_conversation
from .codex_runtime.turn import run_turn
from .codex_runtime.workspace import prepare_workspace
from .eval import EvalRequest, run_eval
from .logging_config import compact_text, configure_logging, log_event
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

_conv_locks: dict[str, asyncio.Lock] = {}


def _lock_for(cid: str) -> asyncio.Lock:
    return _conv_locks.setdefault(cid, asyncio.Lock())


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


_active_browser_turns: dict[str, ActiveBrowserTurn] = {}


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


def _autonomous_for_turn(conv: dict, requested_autonomous: bool) -> bool:
    """Lock autonomous mode once the conversation has a user-visible history."""
    if conv.get("messages"):
        return bool(conv.get("autonomous"))
    return requested_autonomous


class NewConversation(BaseModel):
    sandbox: str = DEFAULT_SANDBOX
    autonomous: bool = False
    # Co-evolution (used by vibe-serve): host path to the caller's candidate
    # workspace to bind read-only at /candidate in this conversation's container.
    peer_workspace: str | None = None
    # Materialize workspaces/<cid>/main eagerly at create (instead of lazily on
    # the first message) so the caller can bind-mount it read-only immediately.
    eager: bool = False


class SendMessage(BaseModel):
    text: str
    sandbox_mode: str = DEFAULT_SANDBOX
    autonomous_mode: bool = False


class AgentSendMessage(BaseModel):
    """One agent turn. `sandbox_mode`/`autonomous_mode` are optional per-turn
    overrides; when omitted they inherit the conversation's create-time settings
    (so a read-only conversation stays read-only unless a turn opts up)."""

    text: str
    sandbox_mode: str | None = None
    autonomous_mode: bool | None = None


@app.get("/")
def index() -> FileResponse:
    dist_index = FRONTEND_DIST / "index.html"
    if dist_index.exists():
        return FileResponse(dist_index)
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/conversations")
def list_conversations() -> dict:
    return {"conversations": store.list(), "sandbox_modes": list(SANDBOX_MODES)}


@app.post("/api/conversations")
def create_conversation(body: NewConversation) -> dict:
    cid = uuid.uuid4().hex[:12]
    fingerprint = prompt_fingerprint(autonomous=body.autonomous)
    log_event(
        LOG,
        "conversation.create",
        conversation_id=cid,
        sandbox=body.sandbox,
        autonomous=body.autonomous,
        prompt_fingerprint=fingerprint,
    )
    return store.create(cid, body.sandbox, fingerprint, autonomous=body.autonomous)


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


# FileNotFoundError/PermissionError/ValueError raised by artifacts.py map to
# 404/403/400 so the agent gets an actionable status without HTTP leaking into
# that module.
def _artifact_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/api/agent/artifacts")
def list_workspace_artifacts(
    cid: str = Query(...),
    subdir: str | None = Query(default=None),
    limit: int = Query(default=2000, ge=1, le=20000),
    _: None = Depends(require_token),
) -> dict:
    try:
        return list_artifacts(cid, subdir=subdir, limit=limit)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise _artifact_http_error(exc) from exc


@app.get("/api/agent/artifacts/download")
def download_workspace_artifact(
    cid: str = Query(...),
    path: str = Query(...),
    _: None = Depends(require_token),
) -> FileResponse:
    try:
        resolved = resolve_artifact(cid, path)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise _artifact_http_error(exc) from exc
    return FileResponse(str(resolved), filename=resolved.name)


# --- Agent conversation API: the real interactive interface -----------------
# Multi-turn, synchronous JSON, token-gated. Reuses the same `store` and
# `run_turn` as the browser SSE path, so a conversation keeps its isolated
# workspace and resumes its orchestrator/implementer Codex sessions across turns.
# The calling agent reads `final` and, like a human, decides whether to answer a
# clarifying question or steer with another turn. Artifacts are fetched via
# /api/agent/artifacts* using the returned `conversation_id` as `cid`.


@app.post("/api/agent/conversations")
def agent_create_conversation(body: NewConversation, _: None = Depends(require_token)) -> dict:
    cid = uuid.uuid4().hex[:12]
    fingerprint = prompt_fingerprint(autonomous=body.autonomous)
    log_event(
        LOG,
        "agent.conversation.create",
        conversation_id=cid,
        sandbox=body.sandbox,
        autonomous=body.autonomous,
        prompt_fingerprint=fingerprint,
        peer_workspace=body.peer_workspace,
        eager=body.eager,
    )
    conv = store.create(
        cid,
        body.sandbox,
        fingerprint,
        autonomous=body.autonomous,
        peer_workspace=body.peer_workspace,
    )
    if body.eager:
        # Materialize the isolated workspace now so the caller can bind-mount it
        # read-only before its own container launches (the container itself is
        # still started lazily on the first message). Return its host path.
        workspace_main = prepare_workspace(cid, autonomous=body.autonomous)
        conv["workspace_path"] = str(workspace_main)
    return conv


@app.get("/api/agent/conversations/{cid}")
def agent_get_conversation(cid: str, _: None = Depends(require_token)) -> dict:
    conv = store.get(cid)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@app.delete("/api/agent/conversations/{cid}")
def agent_delete_conversation(cid: str, _: None = Depends(require_token)) -> dict:
    log_event(LOG, "agent.conversation.delete", conversation_id=cid)
    store.delete(cid)
    _conv_locks.pop(cid, None)
    cleanup_conversation(cid)
    return {"ok": True}


@app.post("/api/agent/conversations/{cid}/messages")
async def agent_send_message(
    cid: str, body: AgentSendMessage, _: None = Depends(require_token)
) -> dict:
    conv = store.get(cid)
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
    fingerprint = prompt_fingerprint(autonomous=autonomous)
    turn_id = uuid.uuid4().hex[:10]
    store.update_runtime_settings(cid, sandbox=sandbox, autonomous=autonomous)
    store.add_message(cid, "user", text)
    sessions = store.sessions_for_prompt(cid, fingerprint)
    log_event(
        LOG,
        "agent.turn.received",
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

    lock = _lock_for(cid)
    async with lock:
        try:
            async for ev in run_turn(
                cid,
                text,
                sandbox=sandbox,
                sessions=sessions,
                turn_id=turn_id,
                prompt_fingerprint=fingerprint,
                autonomous=autonomous,
                peer_dir=conv.get("peer_workspace"),
            ):
                if ev.get("kind") == "session":
                    store.set_codex_session(cid, ev.get("role", ""), ev.get("session_id"))
                collect_turn_event(result, ev)
            result["ok"] = bool(result["final"]) and not bool(result["error"])
        except Exception as exc:  # surface backend failures in-band, like /api/eval
            result["error"] = str(exc)
            LOG.exception(
                "agent.turn.error",
                extra={"event_fields": {"event": "agent.turn.error", "conversation_id": cid, "turn_id": turn_id, "error": str(exc)}},
            )
        if not result["final"]:
            result["final"] = f"(backend error: {result['error']})" if result["error"] else "(no answer)"
        store.add_message(
            cid,
            "assistant",
            result["final"],
            intermediate_outputs=result["intermediate_outputs"] or None,
        )
    log_event(
        LOG,
        "agent.turn.complete",
        conversation_id=cid,
        turn_id=turn_id,
        ok=result["ok"],
        final_len=len(result["final"]),
        final_preview=compact_text(result["final"]),
    )
    return result


@app.get("/api/conversations/{cid}")
def get_conversation(
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
        store.get_message_page(cid, before=before, limit=limit)
        if limit is not None
        else store.get(cid)
    )
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str) -> dict:
    log_event(LOG, "conversation.delete", conversation_id=cid)
    store.delete(cid)
    _conv_locks.pop(cid, None)
    cleanup_conversation(cid)
    return {"ok": True}


# Serve image files the assistant references in markdown (e.g. generated plots under
# ../main/logs/...). Relative paths resolve against ../main (the assistant's workdir).
# Guarded: must stay inside the workspace, and only image extensions are served.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"}


@app.get("/api/file")
def serve_file(path: str = Query(...), cid: str | None = Query(default=None)) -> FileResponse:
    requested = Path(path)
    root = workspace_main_for(cid) if cid else MAIN_DIR
    if requested == Path("/workspace") or Path("/workspace") in requested.parents:
        requested = root / requested.relative_to("/workspace")
    elif not requested.is_absolute():
        requested = root / requested
    try:
        resolved = requested.resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="not found")

    allowed_root = (WORKSPACES_DIR if cid else WORKSPACE).resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise HTTPException(status_code=403, detail="outside workspace")
    if resolved.suffix.lower() not in _IMAGE_EXTS:
        raise HTTPException(status_code=415, detail="unsupported file type")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="not a file")
    return FileResponse(str(resolved))


@app.post("/api/conversations/{cid}/messages")
async def send_message(cid: str, body: SendMessage) -> StreamingResponse:
    conv = store.get(cid)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    active_turn = _active_browser_turns.get(cid)
    if active_turn is not None and not active_turn.finished:
        raise HTTPException(status_code=409, detail="conversation already has an active turn")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty message")

    sandbox = body.sandbox_mode
    autonomous = _autonomous_for_turn(conv, body.autonomous_mode)
    current_prompt_fingerprint = prompt_fingerprint(autonomous=autonomous)
    turn_id = uuid.uuid4().hex[:10]
    previous_sessions = dict(conv.get("codex_sessions") or {})
    store.update_runtime_settings(cid, sandbox=sandbox, autonomous=autonomous)
    store.add_message(cid, "user", text)
    sessions = store.sessions_for_prompt(cid, current_prompt_fingerprint)
    log_event(
        LOG,
        "turn.received",
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
    _active_browser_turns[cid] = active_turn
    active_turn.task = asyncio.create_task(
        _run_browser_turn(
            cid=cid,
            text=text,
            sandbox=sandbox,
            sessions=sessions,
            turn_id=turn_id,
            prompt_fingerprint=current_prompt_fingerprint,
            autonomous=autonomous,
            active_turn=active_turn,
        )
    )
    return _turn_stream_response(active_turn)


@app.get("/api/conversations/{cid}/stream")
async def resume_message_stream(cid: str) -> StreamingResponse:
    if store.get(cid) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    active_turn = _active_browser_turns.get(cid)
    if active_turn is None or active_turn.finished:
        raise HTTPException(status_code=409, detail="conversation has no active turn")
    return _turn_stream_response(active_turn)


@app.post("/api/conversations/{cid}/cancel")
async def cancel_message(cid: str) -> dict:
    if store.get(cid) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    active_turn = _active_browser_turns.get(cid)
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
    cid: str,
    text: str,
    sandbox: str,
    sessions: dict[str, str],
    turn_id: str,
    prompt_fingerprint: str,
    autonomous: bool,
    active_turn: ActiveBrowserTurn,
) -> None:
    lock = _lock_for(cid)
    async with lock:
        final_text: str | None = None
        intermediate_outputs: list[dict[str, str]] = []
        # Persist render-relevant events on completion; retain all SSE events in
        # ActiveBrowserTurn during execution so a refreshed client can replay them.
        activity: list[dict] = []
        try:
            async for ev in run_turn(
                cid,
                text,
                sandbox=sandbox,
                sessions=sessions,
                turn_id=turn_id,
                prompt_fingerprint=prompt_fingerprint,
                autonomous=autonomous,
            ):
                kind = ev.get("kind")
                if kind == "session":
                    store.set_codex_session(cid, ev.get("role", ""), ev.get("session_id"))
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
                            {"role": ev.get("role"), "session_id": ev.get("session_id")},
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
                        "text": str(ev.get("text") or ""),
                    }
                    intermediate_outputs.append(intermediate_output)
                    activity.append({"kind": "intermediate_output", **intermediate_output})
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
                        "duration_ms": int(ev.get("duration_ms") or 0),
                        "tokens": ev.get("tokens") or {},
                    }
                    activity.append({"kind": "usage", **usage})
                    await active_turn.publish(_sse("usage", usage))
                elif kind in ("progress", "error"):
                    await active_turn.publish(
                        _sse("progress", {"text": ev.get("text", "")})
                    )
                elif kind == "final":
                    final_text = ev.get("text") or ""
            if final_text is None:
                final_text = "(no answer)"
            log_event(
                LOG,
                "turn.complete",
                conversation_id=cid,
                turn_id=turn_id,
                final_len=len(final_text),
                final_preview=compact_text(final_text),
            )
        except asyncio.CancelledError:
            final_text = "Stopped."
            log_event(
                LOG,
                "turn.cancelled",
                conversation_id=cid,
                turn_id=turn_id,
            )
        except Exception as exc:  # surface backend failures to the UI
            final_text = f"(backend error: {exc})"
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
                activity.append({"kind": "final", "text": final_text})
                store.add_message(
                    cid,
                    "assistant",
                    final_text,
                    intermediate_outputs=intermediate_outputs or None,
                    activity=activity or None,
                )
                await active_turn.publish(_sse("done", {"text": final_text}))
            await active_turn.finish()
            if _active_browser_turns.get(cid) is active_turn:
                _active_browser_turns.pop(cid, None)


# Mounted last so it doesn't shadow the API routes above. The directory may not
# exist before the first frontend build; `run.sh` creates it for normal serving.
app.mount(
    "/assets",
    StaticFiles(directory=str(FRONTEND_DIST / "assets"), check_dir=False),
    name="assets",
)
