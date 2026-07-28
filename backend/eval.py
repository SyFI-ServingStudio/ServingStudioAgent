"""JSON eval endpoint backed by the normal Codex turn runtime."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from pydantic import BaseModel

from .codex_runtime.config import DEFAULT_SANDBOX, prompt_fingerprint, workspace_main_for
from .codex_runtime.docker import remove_container
from .codex_runtime.turn import run_turn
from .store import WorkspaceRegistry
from .turn_result import collect_turn_event, new_turn_result


class EvalRequest(BaseModel):
    prompt: str
    sandbox: str = DEFAULT_SANDBOX
    autonomous: bool = True
    keep_container: bool = False


async def run_eval(
    *,
    prompt: str,
    sandbox: str = DEFAULT_SANDBOX,
    autonomous: bool = True,
    keep_container: bool = False,
) -> dict[str, Any]:
    """Run one prompt through `run_turn` and return a script-friendly JSON object."""
    text = prompt.strip()
    if not text:
        raise ValueError("empty prompt")

    conversation_id = f"eval-{uuid.uuid4().hex[:12]}"
    workspace_id = f"w_{conversation_id}"
    WorkspaceRegistry().create(
        f"Evaluation {conversation_id[-6:]}",
        workspace_id=workspace_id,
    )
    turn_id = uuid.uuid4().hex[:10]
    fingerprint = prompt_fingerprint(autonomous=autonomous)
    result = new_turn_result(
        conversation_id=conversation_id,
        turn_id=turn_id,
        sandbox=sandbox,
        autonomous=autonomous,
        workspace_id=workspace_id,
        kept_workspace=True,
        kept_container=keep_container,
        workspace=str(workspace_main_for(workspace_id)),
    )

    try:
        async for event in run_turn(
            workspace_id,
            conversation_id,
            text,
            sandbox=sandbox,
            sessions={},
            turn_id=turn_id,
            prompt_fingerprint=fingerprint,
            autonomous=autonomous,
        ):
            collect_turn_event(result, event)
        result["ok"] = bool(result["final"]) and not bool(result["error"])
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if not keep_container:
            await asyncio.to_thread(
                remove_container,
                workspace_id,
                conversation_id,
            )

    return result
