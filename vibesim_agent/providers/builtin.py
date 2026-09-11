"""Explicit built-in profile declarations and stable backend session identities."""

import hashlib
import json
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlsplit

from ..runtime.invocation import InvocationHome
from ..settings import (
    ConfigurationError,
    Settings,
)
from .base import AgentAdapter, Credentials, Model, Provider
from .catalog import FileCatalog
from .registry import ProviderRegistry

CLAUDE_AUTH_ENVIRONMENT = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
CLAUDE_ENVIRONMENT = (*CLAUDE_AUTH_ENVIRONMENT, "ANTHROPIC_BASE_URL")
CLAUDE_MODEL_ALIASES = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-5"}


def provider_adapter(settings: Settings, provider_id: str) -> str:
    connection = settings.connections.get(provider_id)
    if connection is None:
        raise ConfigurationError(f"provider {provider_id} has no connection configuration")
    return connection.adapter


def provider_environment(settings: Settings, provider_id: str) -> dict[str, str]:
    """Resolve only this connection's declared credentials into CLI variable names."""
    connection = settings.connections.get(provider_id)
    if connection is None:
        raise ConfigurationError(f"provider {provider_id} has no connection configuration")
    references = connection.environment
    result = {
        target: settings.secrets[source].get_secret_value()
        for target, source in references.items()
        if source in settings.secrets
    }
    if connection.base_url is not None:
        result["ANTHROPIC_BASE_URL"] = connection.base_url
    return result


def _endpoint(value: str) -> str:
    if not value:
        return ""  # Adapter/CLI built-in endpoint, not a host environment default.
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError(
            "provider endpoint must be an HTTP URL without credentials, query or fragment"
        )
    return value.rstrip("/")


def _codex_backend(home: Path) -> dict:
    try:
        with (home / "config.toml").open("rb") as stream:
            config = tomllib.load(stream)
    except FileNotFoundError:
        config = {}
    except (OSError, ValueError):
        raise ConfigurationError("cannot read Codex backend configuration") from None
    profile = config.get("profile")
    if profile is not None:
        profiles = config.get("profiles", {})
        if (
            not isinstance(profile, str)
            or not isinstance(profiles, dict)
            or not isinstance(profiles.get(profile), dict)
        ):
            raise ConfigurationError("invalid Codex profile configuration")
        config = {**config, **profiles[profile]}
    provider = config.get("model_provider", "openai")
    providers = config.get("model_providers", {})
    if not isinstance(provider, str) or not provider or not isinstance(providers, dict):
        raise ConfigurationError("invalid Codex model provider configuration")
    backend = providers.get(provider, {})
    if not isinstance(backend, dict):
        raise ConfigurationError("invalid Codex backend configuration")
    endpoint = backend.get("base_url", "")
    wire_api = backend.get("wire_api", "responses")
    if not isinstance(endpoint, str) or not isinstance(wire_api, str):
        raise ConfigurationError("invalid Codex backend endpoint or wire API")
    # Only backend identity participates. Auth tokens, retry budgets, selected
    # model/effort, catalog paths and unrelated tool configuration do not.
    return {
        "home": str(home.resolve()),
        "model_provider": provider,
        "base_url": _endpoint(endpoint),
        "wire_api": wire_api,
        "requires_openai_auth": backend.get("requires_openai_auth"),
    }


def session_scope(settings: Settings, provider_id: str, adapter_id: str) -> str:
    selection = settings.providers[provider_id]
    if adapter_id == "codex":
        if selection.home is None:
            raise ConfigurationError("Codex providers require an explicit profile home")
        backend = _codex_backend(selection.home)
    elif adapter_id == "claude":
        backend = {
            "base_url": _endpoint(
                provider_environment(settings, provider_id).get(
                    "ANTHROPIC_BASE_URL", ""
                )
            )
        }
        if selection.home is not None:
            backend["home"] = str(selection.home.resolve())
    else:
        raise ConfigurationError("unsupported built-in adapter")
    identity = {
        "version": 1,
        "provider": provider_id,
        "adapter": adapter_id,
        "backend": backend,
    }
    connection = settings.connections.get(provider_id)
    if connection is not None and connection.session_identity is not None:
        identity["session_identity"] = connection.session_identity
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{adapter_id}:{provider_id}:v1:{digest}"


def guarded_profile_prepare(
    settings: Settings, provider: Provider, prepare: Callable[[InvocationHome], None]
) -> Callable[[InvocationHome], None]:
    """Fail closed if backend identity changed since this registry was built."""

    def validate():
        if (
            session_scope(settings, provider.provider_id, provider.adapter.adapter_id)
            != provider.session_scope
        ):
            raise ConfigurationError(
                "provider backend changed; rebuild the registry before preparing a runtime"
            )

    def guarded(home: InvocationHome) -> None:
        validate()
        prepare(home)
        validate()

    return guarded


def _model(
    model_id: str,
    efforts: tuple[str, ...],
    effort: str,
) -> Model:
    return Model(model_id, model_id, efforts, effort)


def build_registry(
    settings: Settings, *, adapters: Mapping[str, AgentAdapter]
) -> ProviderRegistry:
    registry = ProviderRegistry(settings.secrets)
    for provider_id, selection in settings.providers.items():
        connection = settings.connections.get(provider_id)
        if connection is None or not connection.models:
            raise ConfigurationError(f"provider {provider_id} has no declared models")
        adapter = adapters.get(provider_id)
        expected_adapter = provider_adapter(settings, provider_id)
        if adapter is None or adapter.adapter_id != expected_adapter:
            raise ConfigurationError(
                f"provider {provider_id} requires its {expected_adapter} adapter"
            )
        models = tuple(
            _model(model.model_id, model.efforts, selection.effort)
            for model in connection.models
        )
        if expected_adapter == "claude":
            catalog = FileCatalog(models, ())
            credentials = Credentials(
                any_secrets=tuple(connection.environment.values())
            )
            if selection.home is not None:
                credentials = Credentials(
                    required_files=(selection.home / ".credentials.json",)
                )
            elif not connection.environment:
                raise ConfigurationError(
                    "Claude connections require an authentication source"
                )
        else:
            if selection.home is None:
                raise ConfigurationError(
                    "Codex providers require an explicit profile home"
                )
            credentials = Credentials(
                required_files=(selection.home / "config.toml",),
                all_secrets=tuple(connection.environment.values()),
            )
            catalog = FileCatalog(
                models,
                tuple(
                    selection.home / name
                    for name in ("models_cache.json", "models_catalog.json")
                ),
            )
        registry.register(
            Provider(
                provider_id,
                connection.label or provider_id,
                adapter,
                selection,
                session_scope(settings, provider_id, adapter.adapter_id),
                catalog,
                credentials,
            )
        )
    return registry


def legacy_model_aliases(registry: ProviderRegistry) -> dict[str, str]:
    """Legacy model names for enabled profiles; aliases do not authorize resume."""
    models = {
        provider["id"]: {model["id"] for model in provider["models"]}
        for provider in registry.catalog()
    }
    aliases = {
        alias: model
        for alias, model in CLAUDE_MODEL_ALIASES.items()
        if model in models.get("claude", set())
    }
    for alias, provider_id in (("traditional", "gpt"), ("codexds", "deepseek")):
        if provider_id in models:
            target = registry.provider(provider_id).settings.model
            if target in models[provider_id]:
                aliases[alias] = target
    return aliases
