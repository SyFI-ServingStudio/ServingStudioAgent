"""Shared types for Codex subprocess execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import (
    DEFAULT_CODEX_EFFORT,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_SERVICE_TIER,
    CodexModelSpec,
    codex_model,
)

# UI-facing events are plain dicts keyed by ``kind``. Most values are strings,
# but a few carry richer payloads (e.g. ``usage`` events hold an int duration
# and a nested token-breakdown dict), so the value type is ``Any``.
CodexEvent = dict[str, Any]


@dataclass(frozen=True, slots=True)
class CodexExecRequest:
    container: str
    prompt: str
    label: str
    workspace_id: str
    conversation_id: str
    turn_id: str
    model_id: str = DEFAULT_CODEX_MODEL
    effort: str = DEFAULT_CODEX_EFFORT
    service_tier: str = DEFAULT_CODEX_SERVICE_TIER
    session_id: str | None = None
    output_schema: str | None = None

    @property
    def family_id(self) -> str:
        return codex_model(self.model_id).family_id

    @property
    def model(self) -> CodexModelSpec:
        return codex_model(self.model_id)

    @property
    def is_resume(self) -> bool:
        return self.session_id is not None

    @property
    def schema_arg_used(self) -> bool:
        return bool(self.output_schema)
