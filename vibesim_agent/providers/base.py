"""Provider capabilities and the interface consumed by the turn service."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import SecretStr

from ..domain.roles import Role
from ..settings import ProviderSettings, validate_provider_id


class OutputMode(StrEnum):
    STRUCTURED = "structured"
    PROMPT = "prompt"


@dataclass(frozen=True)
class Model:
    model_id: str
    label: str
    efforts: tuple[str, ...]
    default_effort: str
    service_tiers: tuple[str, ...] = ("default",)
    output_mode: OutputMode = OutputMode.STRUCTURED
    resumable: bool = True

    def __post_init__(self):
        if (
            not self.model_id
            or not self.efforts
            or self.default_effort not in self.efforts
        ):
            raise ValueError("model must have an ID and a supported default effort")
        if not self.service_tiers or "default" not in self.service_tiers:
            raise ValueError("model service tiers must include default")


@dataclass(frozen=True)
class Selection:
    provider_id: str
    model: Model
    effort: str
    service_tier: str
    session_scope: str


@dataclass(frozen=True)
class AgentRequest:
    workspace_id: str
    conversation_id: str
    turn_id: str
    role: Role
    prompt: str
    container: str
    selection: Selection
    session_id: str | None = None
    output_schema: Path | None = None
    execution_id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def structured_output(self) -> bool:
        return (
            self.output_schema is not None
            and self.selection.model.output_mode is OutputMode.STRUCTURED
        )


class AgentAdapter(Protocol):
    adapter_id: str

    def run(self, request: AgentRequest) -> AsyncIterator[dict[str, Any]]:
        """Produce normalized role events; own startup, timeout and cancellation."""
        ...


@dataclass(frozen=True)
class Credentials:
    required_files: tuple[Path, ...] = ()
    all_secrets: tuple[str, ...] = ()
    any_secrets: tuple[str, ...] = ()

    def available(self, secrets: Mapping[str, SecretStr]) -> bool:
        def present(name):
            value = secrets.get(name)
            return value is not None and bool(value.get_secret_value().strip())

        return (
            all(path.is_file() for path in self.required_files)
            and all(present(name) for name in self.all_secrets)
            and (
                not self.any_secrets or any(present(name) for name in self.any_secrets)
            )
        )


@dataclass(frozen=True)
class Provider:
    provider_id: str
    label: str
    adapter: AgentAdapter
    settings: ProviderSettings
    session_scope: str
    catalog: Callable[[], tuple[Model, ...]]
    credentials: Credentials = Credentials()

    def __post_init__(self):
        validate_provider_id(self.provider_id)
        if not self.session_scope:
            raise ValueError("provider identity and session scope are required")
