"""FastAPI app: REST + SSE chat over Docker-backed `codex exec`.

Endpoints:
  GET    /                              -> frontend/index.html
  GET    /static/*                      -> frontend assets
  GET    /api/conversations             -> list (id, title, updated_at)
  POST   /api/conversations             -> create empty conversation
  GET    /api/conversations/{cid}       -> full conversation (messages, role sessions)
  DELETE /api/conversations/{cid}       -> delete
  POST   /api/conversations/{cid}/messages  -> SSE stream of one turn

The message endpoint streams Server-Sent Events: `session` (role Codex session id),
`progress` (transient activity lines), `live_note` (assistant commentary messages),
`orchestrator` (full orchestrator output), `implementer` (implementer summary),
then `done` (stored final answer). `orchestrator` and `implementer` can repeat
inside one browser turn when the orchestrator issues follow-up tasks. A
per-conversation lock prevents two turns racing the same copied
workspace/container.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import codex_runner
from .logging_config import compact_text, configure_logging, log_event
from .store import Store

configure_logging()
LOG = logging.getLogger("mlsim_ui.app")

UI_DIR = Path(__file__).resolve().parents[1]
FRONTEND = UI_DIR / "frontend"

app = FastAPI(title="MLSim Chat")
store = Store()

_conv_locks: dict[str, asyncio.Lock] = {}


def _lock_for(cid: str) -> asyncio.Lock:
    return _conv_locks.setdefault(cid, asyncio.Lock())


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class NewConversation(BaseModel):
    sandbox: str = codex_runner.DEFAULT_SANDBOX


class SendMessage(BaseModel):
    text: str
    sandbox_mode: str = codex_runner.DEFAULT_SANDBOX


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/conversations")
def list_conversations() -> dict:
    return {"conversations": store.list(), "sandbox_modes": list(codex_runner.SANDBOX_MODES)}


@app.post("/api/conversations")
def create_conversation(body: NewConversation) -> dict:
    cid = uuid.uuid4().hex[:12]
    fingerprint = codex_runner.prompt_fingerprint()
    log_event(
        LOG,
        "conversation.create",
        conversation_id=cid,
        sandbox=body.sandbox,
        prompt_fingerprint=fingerprint,
    )
    return store.create(cid, body.sandbox, fingerprint)


@app.get("/api/conversations/{cid}")
def get_conversation(cid: str) -> dict:
    conv = store.get(cid)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str) -> dict:
    log_event(LOG, "conversation.delete", conversation_id=cid)
    store.delete(cid)
    _conv_locks.pop(cid, None)
    codex_runner.cleanup_conversation(cid)
    return {"ok": True}


# Serve image files the assistant references in markdown (e.g. generated plots under
# ../main/logs/...). Relative paths resolve against ../main (the assistant's workdir).
# Guarded: must stay inside the workspace, and only image extensions are served.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"}


@app.get("/api/file")
def serve_file(path: str = Query(...), cid: str | None = Query(default=None)) -> FileResponse:
    requested = Path(path)
    root = codex_runner.workspace_main_for(cid) if cid else codex_runner.MAIN_DIR
    if requested == Path("/workspace") or Path("/workspace") in requested.parents:
        requested = root / requested.relative_to("/workspace")
    elif not requested.is_absolute():
        requested = root / requested
    try:
        resolved = requested.resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="not found")

    allowed_root = (codex_runner.WORKSPACES_DIR if cid else codex_runner.WORKSPACE).resolve()
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

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty message")

    sandbox = body.sandbox_mode
    prompt_fingerprint = codex_runner.prompt_fingerprint()
    turn_id = uuid.uuid4().hex[:10]
    previous_sessions = dict(conv.get("codex_sessions") or {})
    store.add_message(cid, "user", text)
    sessions = store.sessions_for_prompt(cid, prompt_fingerprint)
    log_event(
        LOG,
        "turn.received",
        conversation_id=cid,
        turn_id=turn_id,
        sandbox=sandbox,
        prompt_fingerprint=prompt_fingerprint,
        prompt_len=len(text),
        prompt_preview=compact_text(text),
        previous_session_roles=sorted(previous_sessions),
        active_session_roles=sorted(sessions),
        sessions_reset=bool(previous_sessions) and not sessions,
    )

    async def event_gen():
        lock = _lock_for(cid)
        async with lock:
            final_text: str | None = None
            live_notes: list[dict[str, str]] = []
            try:
                async for ev in codex_runner.run_turn(
                    cid,
                    text,
                    sandbox=sandbox,
                    sessions=sessions,
                    turn_id=turn_id,
                    prompt_fingerprint=prompt_fingerprint,
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
                        yield _sse(
                            "session",
                            {"role": ev.get("role"), "session_id": ev.get("session_id")},
                        )
                    elif kind in ("orchestrator", "implementer"):
                        yield _sse(kind, {"text": ev.get("text", "")})
                    elif kind in ("live_note", "user_progress"):
                        note = {
                            "role": str(ev.get("role") or ""),
                            "text": str(ev.get("text") or ""),
                        }
                        live_notes.append(note)
                        yield _sse("live_note", note)
                    elif kind in ("progress", "error"):
                        yield _sse("progress", {"text": ev.get("text", "")})
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
                store.add_message(cid, "assistant", final_text, live_notes=live_notes or None)
                yield _sse("done", {"text": final_text})
            except asyncio.CancelledError:
                log_event(
                    LOG,
                    "turn.cancelled",
                    conversation_id=cid,
                    turn_id=turn_id,
                )
                raise
            except Exception as exc:  # surface backend failures to the UI
                msg = f"(backend error: {exc})"
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
                store.add_message(cid, "assistant", msg, live_notes=live_notes or None)
                yield _sse("done", {"text": msg})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Mounted last so it doesn't shadow the API routes above.
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
