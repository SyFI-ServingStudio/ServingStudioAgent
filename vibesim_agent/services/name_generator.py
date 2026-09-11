"""Explicit, bounded naming requests with the established model response contract."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel, Field, SecretStr, field_validator

from ..prompts.render import Prompts

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 8.0
SOURCE_TEXT_LIMIT = 6000
SOURCE_TEXT_HEAD = 4500
SOURCE_TEXT_TAIL = 1500


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


class NameGenerator:
    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str,
        base_url: str,
        timeout: float,
        prompts: Prompts,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.prompts = prompts
        self.client_factory = client_factory

    @property
    def enabled(self) -> bool:
        return self.api_key is not None and bool(
            self.api_key.get_secret_value().strip()
        )

    def _prompt_text(self, filename: str) -> str:
        return (self.prompts.directory / filename).read_text(encoding="utf-8").strip()

    async def generate(self, user_message: str, final_answer: str) -> GeneratedNames:
        if not self.enabled:
            raise RuntimeError("OpenRouter naming API key is not configured")
        api_key = self.api_key.get_secret_value().strip()
        model, base_url, timeout_seconds = self.model, self.base_url, self.timeout
        prompt_payload = json.dumps(
            {
                "user_message": _bounded_source_text(user_message),
                "agent_final_answer": _bounded_source_text(final_answer),
            },
            ensure_ascii=False,
        )
        user_prompt = self._prompt_text("naming-user.txt").replace(
            "{payload}", prompt_payload
        )
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
                {"role": "system", "content": self._prompt_text("naming-system.txt")},
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
        async with (
            asyncio.timeout(timeout_seconds),
            self.client_factory(timeout=timeout_seconds) as client,
        ):
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=request_body,
            )
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("OpenRouter naming response content is not text")  # noqa: TRY004 - legacy response contract
        return GeneratedNames.model_validate_json(content)
