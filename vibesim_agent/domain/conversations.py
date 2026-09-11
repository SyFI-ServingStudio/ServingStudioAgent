"""Durable conversation values independent of SQL and provider implementations."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoleRuntime:
    provider_id: str
    session_scope: str
    model_id: str
    effort: str
    service_tier: str


@dataclass(frozen=True)
class Message:
    id: int
    role: str
    content: str
    ts: float
    turn_id: str | None
    metadata: dict[str, Any]
