"""Role selection is independent of model provider and execution environment."""

from enum import StrEnum


class Role(StrEnum):
    ORCHESTRATOR = "orchestrator"
    IMPLEMENTER = "implementer"
    ASSISTANT = "assistant"


class AgentMode(StrEnum):
    ORCHESTRATED = "orchestrated"
    SINGLE = "single"

    @property
    def roles(self) -> tuple[Role, ...]:
        if self is AgentMode.SINGLE:
            return (Role.ASSISTANT,)
        return (Role.ORCHESTRATOR, Role.IMPLEMENTER)

    @property
    def driver(self) -> Role:
        return self.roles[0]


class Sandbox(StrEnum):
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    FULL_ACCESS = "danger-full-access"
