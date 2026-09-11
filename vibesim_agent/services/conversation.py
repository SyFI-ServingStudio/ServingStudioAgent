"""Conversation history projection with stable browser message identities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from uuid import uuid4

from ..domain.conversations import RoleRuntime
from ..domain.errors import ProviderUnavailable
from ..domain.identifiers import validate_conversation_id
from ..domain.roles import AgentMode, Role, Sandbox
from ..providers.registry import ProviderRegistry
from ..storage.registry import WorkspaceRegistry
from .turn import TurnStorage


class UnknownConversationModel(ValueError):
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__("unknown conversation model")


class ConversationRuntimeLocked(ValueError):
    """An existing conversation cannot resume the requested provider scope."""


class ConversationService:
    def __init__(
        self,
        storage: Callable[[str], TurnStorage],
        *,
        providers: ProviderRegistry | None = None,
        default_runtimes: Mapping[Role, RoleRuntime] | None = None,
        fingerprint: Callable[..., str] | None = None,
        model_aliases: Mapping[str, str] | None = None,
        touch_workspace: Callable[[str], object] | None = None,
        workspaces: WorkspaceRegistry | None = None,
    ):
        self.storage = storage
        self.providers = providers
        self.fingerprint = fingerprint
        self.model_aliases = dict(model_aliases or {})
        self.touch_workspace = touch_workspace
        self.workspaces = workspaces
        self.default_runtimes = (
            None if default_runtimes is None else dict(default_runtimes)
        )

    def list(self, workspace_id: str) -> list[dict]:
        return self.storage(workspace_id).conversations.list()

    def list_all(self) -> list[dict]:
        if self.workspaces is None:
            raise ValueError(
                "workspace registry is required for the conversation index"
            )
        records = [
            {**conversation, "workspace_id": workspace["workspace_id"]}
            for workspace in self.workspaces.list()
            for conversation in self.list(workspace["workspace_id"])
        ]
        records.sort(
            key=lambda item: (
                -float(item["updated_at"]),
                item["workspace_id"],
                item["id"],
            )
        )
        return records

    def update_runtime(
        self,
        workspace_id: str,
        conversation_id: str,
        overrides: Mapping[str, Mapping[str, str | None]],
    ) -> dict:
        runtimes = self.browser_runtimes(overrides)
        assert self.providers is not None
        unavailable = tuple(
            sorted(
                {
                    runtime.provider_id
                    for runtime in runtimes.values()
                    if not self.providers.available(runtime.provider_id)
                }
            )
        )
        if unavailable:
            raise ProviderUnavailable(unavailable)
        try:
            self.storage(workspace_id).conversations.update_runtimes(
                conversation_id, runtimes
            )
        except ValueError as error:
            raise ConversationRuntimeLocked(str(error)) from error
        return self.get(workspace_id, conversation_id)

    def browser_runtimes(
        self, overrides: Mapping[str, Mapping[str, str | None]]
    ) -> dict[Role, RoleRuntime]:
        """Resolve optional connection identities while retaining model-only defaults.

        A shared model ID retains the role's default provider when possible;
        otherwise the legacy request has insufficient information to choose one.
        """
        if self.providers is None or self.default_runtimes is None:
            raise ValueError(
                "conversation creation requires configured providers and role runtimes"
            )
        if set(self.default_runtimes) != set(Role):
            raise ValueError("conversation creation requires all role runtimes")
        catalog = self.providers.catalog()
        runtimes = {}
        for role in Role:
            default = self.default_runtimes[role]
            requested = overrides.get(role.value, {})
            explicit_provider = requested.get("provider")
            selection_defaults = (
                self.providers.provider(explicit_provider).settings
                if explicit_provider is not None
                else None
            )
            model_id = requested.get("model") or (
                selection_defaults.model
                if selection_defaults is not None
                else default.model_id
            )
            model_id = self.model_aliases.get(model_id, model_id)
            candidates = {
                provider["id"]: model
                for provider in catalog
                for model in provider["models"]
                if model["id"] == model_id
            }
            if not candidates:
                raise UnknownConversationModel(model_id)
            if explicit_provider is not None:
                if explicit_provider not in candidates:
                    raise ValueError("model is not offered by the requested provider")
                provider_id = explicit_provider
            elif default.provider_id in candidates:
                provider_id = default.provider_id
            elif len(candidates) == 1:
                provider_id = next(iter(candidates))
            else:
                raise ValueError(f"ambiguous provider for model: {model_id}")
            effort = requested.get(
                "effort",
                selection_defaults.effort
                if selection_defaults is not None
                else default.effort,
            )
            tier = requested.get(
                "service_tier",
                selection_defaults.service_tier
                if selection_defaults is not None
                else default.service_tier,
            )
            selected = self.providers.select(
                provider_id,
                model_id,
                effort=effort,
                service_tier=tier,
            )
            runtimes[role] = RoleRuntime(
                provider_id,
                selected.session_scope,
                model_id,
                selected.effort,
                selected.service_tier,
            )
        return runtimes

    def runtime_projection(self, runtimes: Mapping[Role, RoleRuntime]) -> dict:
        owners: dict[str, set[str]] = {}
        if self.providers is not None:
            for provider in self.providers.catalog():
                for model in provider["models"]:
                    owners.setdefault(model["id"], set()).add(provider["id"])
        return {
            role.value: {
                "model": runtime.model_id,
                "effort": runtime.effort,
                "serviceTier": runtime.service_tier,
                **(
                    {"provider": runtime.provider_id}
                    if owners.get(runtime.model_id) != {runtime.provider_id}
                    else {}
                ),
            }
            for role, runtime in runtimes.items()
        }

    def create(
        self,
        workspace_id: str,
        *,
        mode: AgentMode,
        sandbox: Sandbox,
        autonomous: bool,
        peer_workspace: str | None,
        runtimes: Mapping[Role, RoleRuntime] | None = None,
        prompt_fingerprint: str | None = None,
        conversation_id: str | None = None,
    ) -> dict:
        if self.providers is None:
            raise ValueError("conversation creation requires a provider registry")
        configured = self.default_runtimes if runtimes is None else runtimes
        if configured is None:
            raise ValueError("conversation creation requires configured role runtimes")
        resolved = {Role(role): runtime for role, runtime in configured.items()}
        if set(resolved) != set(Role):
            raise ValueError("conversation creation requires all role runtimes")
        mode, sandbox = AgentMode(mode), Sandbox(sandbox)
        for runtime in resolved.values():
            if not runtime.model_id or not runtime.effort or not runtime.service_tier:
                raise ValueError("explicit model, effort and service tier are required")
            selection = self.providers.select(
                runtime.provider_id,
                runtime.model_id,
                effort=runtime.effort,
                service_tier=runtime.service_tier,
            )
            if selection.session_scope != runtime.session_scope:
                raise ValueError("provider session scope changed")
        unavailable = tuple(
            sorted(
                {
                    resolved[role].provider_id
                    for role in mode.roles
                    if not self.providers.available(resolved[role].provider_id)
                }
            )
        )
        if unavailable:
            raise ProviderUnavailable(unavailable)
        if prompt_fingerprint is None and self.fingerprint is not None:
            prompt_fingerprint = self.fingerprint(
                mode=mode, autonomous=autonomous, runtimes=resolved
            )
        conversation_id = (
            uuid4().hex[:12] if conversation_id is None else conversation_id
        )
        validate_conversation_id(conversation_id)
        self.storage(workspace_id).conversations.create(
            conversation_id,
            runtimes=resolved,
            agent_mode=mode,
            sandbox=sandbox,
            autonomous=autonomous,
            peer_workspace=peer_workspace,
            prompt_fingerprint=prompt_fingerprint,
            title="New chat",
            naming_state="pending",
        )
        if self.touch_workspace is not None:
            self.touch_workspace(workspace_id)
        return self.get(workspace_id, conversation_id)

    def get(
        self,
        workspace_id: str,
        conversation_id: str,
        *,
        limit: int | None = None,
        before: int | None = None,
    ) -> dict:
        if before is not None and limit is None:
            raise ValueError("before requires limit")
        store = self.storage(workspace_id)
        conversation = store.conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        if limit is None:
            messages = store.conversations.messages(conversation_id)
            page = None
        else:
            messages, page = store.conversations.page(
                conversation_id, limit=limit, before=before
            )
        result = {
            **conversation,
            "autonomous": bool(conversation["autonomous"]),
            "codex_runtime": self.runtime_projection(
                store.conversations.runtimes(conversation_id)
            ),
            "codex_sessions": {
                session.role.value: session.session_id
                for session in store.sessions.list(conversation_id)
            },
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "ts": message.ts,
                    **message.metadata,
                    "id": message.id,
                    "turn_id": message.turn_id,
                }
                for message in messages
            ],
        }
        if page is not None:
            result["message_page"] = page
        return result
