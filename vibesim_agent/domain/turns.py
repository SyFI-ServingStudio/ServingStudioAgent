"""Turn requests and semantic outcomes independent of HTTP and SQL."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .conversations import RoleRuntime
from .roles import AgentMode, Role, Sandbox


class Outcome(StrEnum):
    ANSWER = "final_answer"
    INPUT = "request_user_input"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TurnOptions:
    sandbox: Sandbox | None = None
    autonomous: bool | None = None
    mode: AgentMode | None = None
    resume_role: Role | str | None = None
    analyzer_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class TurnInput:
    workspace_id: str
    conversation_id: str
    turn_id: str
    text: str
    mode: AgentMode
    runtimes: dict[Role, RoleRuntime]
    sessions: dict[Role, str]
    resume_role: str
    sandbox: Sandbox = Sandbox.WORKSPACE_WRITE
    autonomous: bool = False
    peer_workspace: str | None = None
    prompt_fingerprint: str | None = None
    analyzer_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class TurnResult:
    outcome: Outcome
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    resume_role: Role | None = None
