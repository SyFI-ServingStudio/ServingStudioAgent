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
    ProviderEnvironment,
    ProviderSettings,
    Settings,
)
from .base import AgentAdapter, Credentials, Model, OutputMode, Provider
from .catalog import FileCatalog
from .registry import ProviderRegistry

CLAUDE_AUTH_ENVIRONMENT = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
CLAUDE_ENVIRONMENT = (*CLAUDE_AUTH_ENVIRONMENT, "ANTHROPIC_BASE_URL")
CLAUDE_MODEL_ALIASES = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-5"}
CLAUDE_MODELS = {
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-opus-5": "Claude Opus 5",
}
GPT_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
GPT_EFFORTS = ("low", "medium", "high", "xhigh")
DEEPSEEK_EFFORTS = ("high", "xhigh", "max")
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def provider_environments(host_home: Path) -> tuple[ProviderEnvironment, ...]:
    if not host_home.is_absolute():
        raise ConfigurationError("provider host home must be absolute")
    return (
        ProviderEnvironment(
            "gpt",
            ProviderSettings(
                model="gpt-5.6-sol", effort="xhigh", home=host_home / ".codex"
            ),
            accepts_home=True,
        ),
        ProviderEnvironment(
            "deepseek",
            ProviderSettings(
                model="deepseek-ai/DeepSeek-V4-Flash-0731",
                effort="max",
                home=host_home / ".codex-ds",
            ),
            accepts_home=True,
            secret_names=("VLLM_API_KEY",),
        ),
        ProviderEnvironment(
            "claude",
            ProviderSettings(model="claude-sonnet-5", effort="high"),
            secret_names=CLAUDE_ENVIRONMENT,
        ),
    )


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
        endpoint = settings.secrets.get("ANTHROPIC_BASE_URL")
        backend = {
            "base_url": _endpoint(endpoint.get_secret_value() if endpoint else "")
        }
    else:
        raise ConfigurationError("unsupported built-in adapter")
    identity = {
        "version": 1,
        "provider": provider_id,
        "adapter": adapter_id,
        "backend": backend,
    }
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
    label: str,
    efforts: tuple[str, ...],
    effort: str,
    output_mode: OutputMode = OutputMode.STRUCTURED,
) -> Model:
    return Model(
        model_id,
        label,
        efforts,
        effort if effort in efforts else efforts[-1],
        output_mode=output_mode,
    )


def build_registry(
    settings: Settings, *, adapters: Mapping[str, AgentAdapter]
) -> ProviderRegistry:
    registry = ProviderRegistry(settings.secrets)
    for provider_id, selection in settings.providers.items():
        if provider_id not in {"gpt", "deepseek", "claude"}:
            raise ConfigurationError(
                "unknown built-in provider; register custom providers explicitly"
            )
        adapter = adapters.get(provider_id)
        expected_adapter = "claude" if provider_id == "claude" else "codex"
        if adapter is None or adapter.adapter_id != expected_adapter:
            raise ConfigurationError(
                f"provider {provider_id} requires its {expected_adapter} adapter"
            )
        if provider_id == "claude":
            selection = selection.model_copy(
                update={
                    "model": CLAUDE_MODEL_ALIASES.get(selection.model, selection.model)
                }
            )
            models = tuple(
                _model(
                    model_id,
                    CLAUDE_MODELS.get(model_id, model_id),
                    CLAUDE_EFFORTS
                    if model_id in CLAUDE_MODELS
                    else ("low", "medium", "high"),
                    selection.effort,
                )
                for model_id in dict.fromkeys((selection.model, *CLAUDE_MODELS))
            )
            catalog = FileCatalog(models, (), default_effort=selection.effort)
            credentials = Credentials(any_secrets=CLAUDE_AUTH_ENVIRONMENT)
            label = "Claude"
        else:
            if selection.home is None:
                raise ConfigurationError(
                    "Codex providers require an explicit profile home"
                )
            if provider_id == "gpt":
                models = tuple(
                    _model(model_id, model_id, GPT_EFFORTS, selection.effort)
                    for model_id in GPT_MODELS
                )
                filenames = ("models_cache.json", "models_catalog.json")
                credentials = Credentials(
                    required_files=(selection.home / "config.toml",)
                )
                label = "GPT-5.6"
            else:
                models = (
                    _model(
                        selection.model,
                        selection.model,
                        DEEPSEEK_EFFORTS,
                        selection.effort,
                        OutputMode.PROMPT,
                    ),
                )
                filenames = ("models_catalog.json", "models_cache.json")
                credentials = Credentials(
                    required_files=(selection.home / "config.toml",),
                    all_secrets=("VLLM_API_KEY",),
                )
                label = "DeepSeek"
            catalog = FileCatalog(
                models,
                tuple(selection.home / name for name in filenames),
                default_effort=selection.effort,
            )
        registry.register(
            Provider(
                provider_id,
                label,
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
