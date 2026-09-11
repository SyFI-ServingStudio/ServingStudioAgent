"""Construct real built-in adapters and their matching runtime declarations."""

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import replace

from .application import ProviderSetup
from .domain.conversations import RoleRuntime
from .domain.roles import Role
from .prompts.render import Prompts
from .providers.base import AgentRequest
from .providers.builtin import (
    CLAUDE_ENVIRONMENT,
    build_registry,
    guarded_profile_prepare,
    legacy_model_aliases,
    provider_adapter,
    provider_environment,
)
from .providers.claude.adapter import ClaudeAdapter
from .providers.claude.command import ClaudeCommand
from .providers.claude.home import ClaudeProfile
from .providers.codex.adapter import CodexAdapter
from .providers.codex.command import CodexCommand
from .providers.codex.home import CodexProfile
from .runtime.command import ExecutionEnvironment
from .runtime.invocation import InvocationHome
from .services.runtime import ProviderRuntime
from .settings import ConfigurationError, Settings


def build_builtin_setup(
    settings: Settings,
    home: Callable[[AgentRequest], InvocationHome],
    environment: ExecutionEnvironment,
    prompts: Prompts,
    *,
    host_environment: Mapping[str, str],
    role_providers: Mapping[Role, str] | None = None,
) -> ProviderSetup:
    adapters, profiles, binaries = {}, {}, {}
    credential_names = {
        "VLLM_API_KEY",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        *CLAUDE_ENVIRONMENT,
    }
    credential_names.update(settings.secrets)
    for connection in settings.connections.values():
        credential_names.update(connection.environment)
        credential_names.update(connection.environment.values())
    for provider_id, selection in settings.providers.items():
        adapter_id = provider_adapter(settings, provider_id)
        selected_environment = provider_environment(settings, provider_id)
        inherited = tuple(selected_environment)
        process_environment = {
            key: value
            for key, value in host_environment.items()
            if key not in credential_names
        }
        process_environment.update(selected_environment)
        log = logging.getLogger(f"vibesim_agent.providers.{provider_id}")
        if adapter_id == "claude":
            schemas = {
                f"{role.value}.schema.json": json.loads(
                    prompts.schema_path(role).read_text()
                )
                for role in Role
            }
            adapters[provider_id] = ClaudeAdapter(
                ClaudeCommand(environment, schemas, inherited_environment=inherited),
                home=home,
                idle_timeout=settings.agent.idle_timeout,
                logger=log,
                process_environment=process_environment,
            )
            profiles[provider_id] = ClaudeProfile(selection.home)
            binaries[provider_id] = ("claude",)
        elif adapter_id == "codex":
            if selection.home is None:
                raise ConfigurationError(
                    "Codex providers require an explicit profile home"
                )
            command = CodexCommand(environment, inherited_environment=inherited)
            adapters[provider_id] = CodexAdapter(
                command,
                home=home,
                idle_timeout=settings.agent.idle_timeout,
                logger=log,
                command_for_home=lambda target, command=command: replace(
                    command,
                    catalog_filename="models_catalog.json"
                    if (target.host / "models_catalog.json").is_file()
                    else None,
                ),
                process_environment=process_environment,
            )
            profiles[provider_id] = CodexProfile(selection.home)
            binaries[provider_id] = ("codex",)
        else:
            raise ConfigurationError(
                "unknown built-in provider; use an explicit provider factory"
            )
    registry = build_registry(settings, adapters=adapters)
    runtimes = {
        provider_id: ProviderRuntime(
            guarded_profile_prepare(
                settings, registry.provider(provider_id), profile.prepare
            ),
            binaries[provider_id],
        )
        for provider_id, profile in profiles.items()
    }
    role_providers = (
        dict(role_providers)
        if role_providers is not None
        else settings.role_providers or {role: "gpt" for role in Role}
    )
    if set(role_providers) != set(Role):
        raise ConfigurationError("all role provider defaults must be configured")
    defaults = {}
    for role, provider_id in role_providers.items():
        selected = registry.select(provider_id)
        defaults[Role(role)] = RoleRuntime(
            provider_id,
            selected.session_scope,
            selected.model.model_id,
            selected.effort,
            selected.service_tier,
        )
    docker_environment = {
        key: value
        for key, value in host_environment.items()
        if key not in credential_names
    }
    return ProviderSetup(
        registry, runtimes, defaults, legacy_model_aliases(registry), docker_environment
    )
