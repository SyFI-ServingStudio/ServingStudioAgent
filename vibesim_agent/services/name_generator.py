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
# The branch is asked for while the reader waits on "Send", so it gets far less
# time than naming after an answer; missing it only costs a plainer name.
BRANCH_TIMEOUT_SECONDS = 4.0
BRANCH_MAX_LENGTH = 40


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

    async def branch_topic(self, request: str) -> str:
        """A short kebab-case topic for the branch a new worktree will own.

        Asked before the worktree exists, from the question alone: the branch
        cannot be renamed later the way a display name can, so it gets the
        model's reading of the request rather than its first forty characters.
        The caller slugs the answer again, so a reply that ignores the format
        can only ever produce a plainer name, never an invalid ref.
        """
        self._require_enabled()
        payload = json.dumps({"request": _bounded_source_text(request)}, ensure_ascii=False)
        content = await self._complete(
            system=self._prompt_text("branch-system.txt"),
            user=self._prompt_text("branch-user.txt").replace("{payload}", payload),
            name="servingstudio_branch",
            schema={
                "type": "object",
                "properties": {
                    "branch": {"type": "string", "minLength": 3, "maxLength": BRANCH_MAX_LENGTH}
                },
                "required": ["branch"],
                "additionalProperties": False,
            },
            max_tokens=96,
            timeout_seconds=min(self.timeout, BRANCH_TIMEOUT_SECONDS),
            # Some providers behind the router think first, and a few words of
            # output leave no room for that: the budget went on reasoning and
            # the content came back empty on 9 of 15 measured calls.
            reasoning=False,
        )
        branch = json.loads(content)["branch"]
        if not isinstance(branch, str) or not branch.strip():
            raise ValueError("naming response has no branch")
        return branch.strip()

    async def generate(self, user_message: str, final_answer: str) -> GeneratedNames:
        self._require_enabled()
        prompt_payload = json.dumps(
            {
                "user_message": _bounded_source_text(user_message),
                "agent_final_answer": _bounded_source_text(final_answer),
            },
            ensure_ascii=False,
        )
        content = await self._complete(
            system=self._prompt_text("naming-system.txt"),
            user=self._prompt_text("naming-user.txt").replace("{payload}", prompt_payload),
            name="vibesim_names",
            schema={
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
            },
            max_tokens=96,
            timeout_seconds=self.timeout,
            # The same budget as the branch topic, lost the same way: with
            # reasoning on, the content came back empty and naming failed.
            reasoning=False,
        )
        return GeneratedNames.model_validate_json(content)

    def _require_enabled(self) -> None:
        # Before any prompt is read: a disabled generator touches nothing.
        if not self.enabled:
            raise RuntimeError("OpenRouter naming API key is not configured")

    async def _complete(
        self,
        *,
        system: str,
        user: str,
        name: str,
        schema: dict[str, Any],
        max_tokens: int,
        timeout_seconds: float,
        reasoning: bool | None = None,
    ) -> str:
        api_key = self.api_key.get_secret_value().strip()
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
            "provider": {
                "require_parameters": True,
                "zdr": True,
                "allow_fallbacks": True,
            },
        }
        if reasoning is not None:
            request_body["reasoning"] = {"enabled": reasoning}
        async with (
            asyncio.timeout(timeout_seconds),
            self.client_factory(timeout=timeout_seconds) as client,
        ):
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=request_body,
            )
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("OpenRouter naming response content is not text")  # noqa: TRY004 - legacy response contract
        return content
