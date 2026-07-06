"""Shared types for Codex subprocess execution."""

from __future__ import annotations

from dataclasses import dataclass

CodexEvent = dict[str, str]


@dataclass(frozen=True, slots=True)
class CodexExecRequest:
    container: str
    prompt: str
    label: str
    conversation_id: str
    turn_id: str
    session_id: str | None = None
    output_schema: str | None = None

    @property
    def is_resume(self) -> bool:
        return self.session_id is not None

    @property
    def schema_arg_used(self) -> bool:
        return bool(self.output_schema and not self.is_resume)
