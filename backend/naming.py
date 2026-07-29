"""Best-effort OpenRouter naming for newly created managed UI objects.

Naming never blocks a turn response.  The generated values are committed through
store-level compare-and-set operations, so a late model response cannot overwrite
a manual rename or a name produced by another task.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from .store import Store

LOG = logging.getLogger("vibesim_ui.naming")
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 8.0
SOURCE_TEXT_LIMIT = 6000
SOURCE_TEXT_HEAD = 4500
SOURCE_TEXT_TAIL = 1500

_background_tasks: set[asyncio.Task[None]] = set()


class GeneratedNames(BaseModel):
    workspace_name: str = Field(min_length=3, max_length=56)
    conversation_title: str = Field(min_length=3, max_length=48)

    @field_validator("workspace_name", "conversation_title")
    @classmethod
    def clean_name(cls, value: str) -> str:
        clean_value = " ".join(value.strip().splitlines()).strip()
        if not clean_value:
            raise ValueError("generated name must not be empty")
        return clean_value


def _bounded_source_text(value: str) -> str:
    if len(value) <= SOURCE_TEXT_LIMIT:
        return value
    return f"{value[:SOURCE_TEXT_HEAD]}\n…\n{value[-SOURCE_TEXT_TAIL:]}"


def _naming_config() -> tuple[str, str, str, float] | None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.environ.get("VIBESIM_NAMING_MODEL", DEFAULT_MODEL).strip()
    base_url = os.environ.get("VIBESIM_NAMING_BASE_URL", DEFAULT_BASE_URL).strip()
    timeout_text = os.environ.get(
        "VIBESIM_NAMING_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)
    ).strip()
    try:
        timeout_seconds = float(timeout_text)
    except ValueError:
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    if timeout_seconds <= 0:
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    return (
        api_key,
        model or DEFAULT_MODEL,
        (base_url or DEFAULT_BASE_URL).rstrip("/"),
        timeout_seconds,
    )


def _prompt_text(filename: str) -> str:
    return (PROMPT_DIR / filename).read_text("utf-8").strip()


async def generate_names(user_message: str, final_answer: str) -> GeneratedNames:
    config = _naming_config()
    if config is None:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    api_key, model, base_url, timeout_seconds = config
    prompt_payload = json.dumps(
        {
            "user_message": _bounded_source_text(user_message),
            "agent_final_answer": _bounded_source_text(final_answer),
        },
        ensure_ascii=False,
    )
    user_prompt = _prompt_text("naming-user.txt").replace("{payload}", prompt_payload)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "workspace_name": {"type": "string", "minLength": 3, "maxLength": 56},
            "conversation_title": {
                "type": "string",
                "minLength": 3,
                "maxLength": 48,
            },
        },
        "required": ["workspace_name", "conversation_title"],
        "additionalProperties": False,
    }
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _prompt_text("naming-system.txt")},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 96,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "vibesim_names",
                "strict": True,
                "schema": schema,
            },
        },
        "provider": {
            "require_parameters": True,
            "zdr": True,
            "allow_fallbacks": True,
        },
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=request_body,
        )
        response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("OpenRouter naming response content is not text")
    return GeneratedNames.model_validate_json(content)


async def _run_auto_naming(
    store: Store,
    workspace_id: str,
    conversation_id: str,
    user_message: str,
    final_answer: str,
) -> None:
    try:
        names = await generate_names(user_message, final_answer)
        workspace_updated = store.registry.apply_generated_name(
            workspace_id, names.workspace_name
        )
        conversation_updated = store.apply_generated_conversation_title(
            workspace_id,
            conversation_id,
            names.conversation_title,
        )
        LOG.info(
            "auto naming complete",
            extra={
                "event_fields": {
                    "event": "naming.complete",
                    "workspace_id": workspace_id,
                    "conversation_id": conversation_id,
                    "workspace_updated": workspace_updated,
                    "conversation_updated": conversation_updated,
                }
            },
        )
    except Exception as exc:
        # Keep pending state so the next successful turn can retry.
        LOG.warning(
            "auto naming failed",
            extra={
                "event_fields": {
                    "event": "naming.failed",
                    "workspace_id": workspace_id,
                    "conversation_id": conversation_id,
                    "error": str(exc),
                }
            },
        )


def schedule_auto_naming(
    store: Store,
    workspace_id: str,
    conversation_id: str,
    user_message: str,
    final_answer: str,
) -> bool:
    """Schedule naming only when a target is pending and OpenRouter is configured."""

    if _naming_config() is None:
        return False
    workspace_pending = store.registry.get(workspace_id)["naming_state"] == "pending"
    conversation_pending = (
        store.conversation_naming_state(workspace_id, conversation_id) == "pending"
    )
    if not workspace_pending and not conversation_pending:
        return False
    task = asyncio.create_task(
        _run_auto_naming(
            store,
            workspace_id,
            conversation_id,
            user_message,
            final_answer,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return True
