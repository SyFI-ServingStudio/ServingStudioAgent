"""Explicit configuration loading, validation, and environment documentation.

No settings are loaded at import. Application and tool composition roots pass
the resulting objects to their owners; domain and storage never read os.environ.
"""

from __future__ import annotations

import os
import pwd
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)


class ConfigurationError(ValueError):
    """Invalid configuration; messages identify keys without exposing values."""


class ConfigModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class AgentSettings(ConfigModel):
    repo_root: Path = Field(
        description="Agent source checkout", json_schema_extra={"env": False}
    )
    main_dir: Path = Field(description="Source VibeSim checkout")
    workspaces_root: Path = Field(description="Workspace registry and durable state")
    bind: str = Field(
        default="127.0.0.1", min_length=1, description="HTTP bind address"
    )
    port: int = Field(default=8765, ge=1, le=65535, description="HTTP listen port")
    api_token: SecretStr = Field(
        default=SecretStr(""), description="Token for tools API; empty opens it"
    )
    idle_timeout: float = Field(
        default=600, gt=0, description="CLI idle timeout in seconds"
    )
    safe_interrupt_timeout: float = Field(
        default=30, gt=0, description="Wait for a recoverable role before stopping"
    )
    analyzer_source: Literal["external", "workspace"] = Field(
        default="external", description="Default Analyzer source"
    )
    analyzer_base_url: str = Field(
        default="http://host.docker.internal:8787",
        description="Analyzer URL visible to runner",
    )
    managed_backend_url: str = Field(
        default="http://host.docker.internal:8765",
        description="Callback URL visible to runner",
    )
    naming_model: str = Field(
        default="deepseek/deepseek-v4-flash",
        min_length=1,
        description="Automatic naming model",
    )
    naming_base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="Automatic naming endpoint"
    )
    naming_timeout: float = Field(
        default=8, gt=0, description="Automatic naming timeout in seconds"
    )

    @field_validator("analyzer_base_url", "managed_backend_url", "naming_base_url")
    @classmethod
    def http_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("must be an absolute HTTP URL")
        if parsed.username or parsed.password or parsed.fragment or parsed.query:
            raise ValueError("must not contain credentials, query, or fragment")
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError("invalid port")
        return value.rstrip("/")

    @field_validator("analyzer_base_url")
    @classmethod
    def analyzer_origin(cls, value: str) -> str:
        if urlsplit(value).path:
            raise ValueError("must be an HTTP origin without a path")
        return value

    @field_validator("analyzer_source", mode="before")
    @classmethod
    def normalize_analyzer_source(cls, value):
        if isinstance(value, str):
            value = value.strip().lower()
        return {"host": "external", "local": "workspace"}.get(value, value)

    @field_validator("repo_root", "main_dir", "workspaces_root")
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("must be an absolute path")
        return value


class ContainerSettings(ConfigModel):
    image: str = Field(min_length=1, description="Runner image tag")
    uid: int = Field(ge=0, description="Container user ID")
    gid: int = Field(ge=0, description="Container group ID")
    user: str = Field(min_length=1, description="Container user name")
    home: Path = Field(description="Home path inside the container")
    gpus: str = Field(
        default="all", description="Docker GPU selection; empty disables GPU request"
    )
    version: str = Field(
        default="prebuilt-agent-runner-v12",
        min_length=1,
        description="Expected runner image version",
    )
    uv_project_environment: Path = Field(
        default=Path("/opt/vibesim-venv"),
        description="Baked Python environment inside runner",
    )
    uv_cache_dir: Path = Field(
        default=Path("/opt/vibesim-uv-cache"), description="uv cache inside runner"
    )
    dg_use_local_version: bool = Field(
        default=False, description="Use a local DeepGEMM build"
    )
    hf_home: Path | None = Field(
        default=None,
        description="Host Hugging Face cache mounted read-only",
        json_schema_extra={"env": "HF_HOME"},
    )

    @field_validator("home", "uv_project_environment", "uv_cache_dir", "hf_home")
    @classmethod
    def absolute_path(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("must be an absolute path")
        return value


class ProviderSettings(ConfigModel):
    model: str = Field(min_length=1, description="Default model ID")
    effort: str = Field(
        min_length=1,
        description="Default reasoning effort, checked against model capabilities",
    )
    service_tier: str = Field(
        default="default", min_length=1, description="Default service tier"
    )
    home: Path | None = Field(
        default=None,
        description="Host provider configuration source, not conversation history",
    )

    @field_validator("home")
    @classmethod
    def absolute_home(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("must be an absolute path")
        return value


class ConnectionSettings(ConfigModel):
    adapter: Literal["codex", "claude"]
    label: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    base_url: str | None = None
    session_identity: str | None = None


def validate_provider_id(provider_id: str) -> str:
    if re.fullmatch(r"[a-z][a-z0-9_]*", provider_id) is None:
        raise ConfigurationError(
            "provider_id must contain lowercase letters, digits or underscores"
        )
    return provider_id


@dataclass(frozen=True)
class ProviderEnvironment:
    """A registered provider declares its configuration needs to the loader."""

    provider_id: str
    defaults: ProviderSettings
    accepts_home: bool = False
    secret_names: tuple[str, ...] = ()

    @property
    def prefix(self) -> str:
        return f"VIBESIM_PROVIDER_{validate_provider_id(self.provider_id).upper()}_"


class Settings(ConfigModel):
    agent: AgentSettings
    container: ContainerSettings
    providers: dict[str, ProviderSettings]
    secrets: dict[str, SecretStr] = Field(default_factory=dict, repr=False)
    connections: dict[str, ConnectionSettings] = Field(default_factory=dict)
    role_providers: dict[str, str] = Field(default_factory=dict)


def _environment_name(prefix: str, name: str, field) -> str | None:
    override = (field.json_schema_extra or {}).get("env")
    if override is False:
        return None
    return override or f"{prefix}{name.upper()}"


def _load(
    model,
    prefix: str,
    environment: Mapping[str, str],
    defaults: dict,
    *,
    host_paths: tuple[str, ...] = (),
):
    values = dict(defaults)
    for name, field in model.model_fields.items():
        key = _environment_name(prefix, name, field)
        if key is not None and key in environment:
            value = environment[key].strip()
            if name == "hf_home" and not value:
                values[name] = None
            else:
                values[name] = value
        if name in host_paths and values.get(name) is not None:
            path = Path(values[name])
            try:
                if path.parts and path.parts[0] == "~":
                    home = environment.get("HOME") or pwd.getpwuid(os.getuid()).pw_dir
                    path = Path(home).joinpath(*path.parts[1:])
                else:
                    path = path.expanduser()
            except (RuntimeError, KeyError):
                raise ConfigurationError(
                    f"Invalid configuration: {key or name}"
                ) from None
            values[name] = path
    try:
        return model.model_validate(values)
    except ValidationError as error:
        keys = sorted(
            {
                _environment_name(
                    prefix, str(item["loc"][0]), model.model_fields[str(item["loc"][0])]
                )
                or str(item["loc"][0])
                for item in error.errors()
            }
        )
        raise ConfigurationError("Invalid configuration: " + ", ".join(keys)) from None


def load_settings(
    *,
    environment: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
    providers: Sequence[ProviderEnvironment] = (),
) -> Settings:
    environment = dict(os.environ if environment is None else environment)
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    user = (
        environment.get("VIBESIM_RUNNER_USER")
        or environment.get("USER")
        or pwd.getpwuid(os.getuid()).pw_name
    ).strip()
    image_owner = re.sub(r"[^a-z0-9_.-]+", "-", user.lower()).strip("-.") or "agent"
    agent = _load(
        AgentSettings,
        "VIBESIM_AGENT_",
        environment,
        {
            "repo_root": root,
            "main_dir": root.parent / "VibeSim",
            "workspaces_root": root.parent / "agent-workspaces",
        },
        host_paths=("main_dir", "workspaces_root"),
    )
    container = _load(
        ContainerSettings,
        "VIBESIM_RUNNER_",
        environment,
        {
            "image": f"vibesim-agent-runner:{image_owner}",
            "uid": os.getuid(),
            "gid": os.getgid(),
            "user": user,
            "home": Path(f"/home/{user}"),
        },
        host_paths=("hf_home",),
    )
    if "VIBESIM_AGENT_PROVIDERS_FILE" in environment:
        if any(name.startswith("VIBESIM_PROVIDER_") for name in environment):
            raise ConfigurationError(
                "Providers file conflicts with VIBESIM_PROVIDER_* overrides"
            )
        from .provider_config import load_provider_config

        configured = load_provider_config(
            Path(environment["VIBESIM_AGENT_PROVIDERS_FILE"]), environment=environment
        )
        secrets = dict(configured.secrets)
        if environment.get("OPENROUTER_API_KEY", "").strip():
            secrets["OPENROUTER_API_KEY"] = SecretStr(
                environment["OPENROUTER_API_KEY"].strip()
            )
        return Settings(
            agent=agent,
            container=container,
            providers=configured.providers,
            secrets=secrets,
            connections=configured.connections,
            role_providers=configured.role_providers,
        )
    selections = {}
    secret_names = {"OPENROUTER_API_KEY"}
    for provider in providers:
        prefix = provider.prefix
        if provider.provider_id in selections:
            raise ConfigurationError(f"Duplicate provider: {provider.provider_id}")
        if not provider.accepts_home and (
            prefix + "HOME" in environment or provider.defaults.home is not None
        ):
            raise ConfigurationError(f"{prefix}HOME is not supported by this provider")
        selections[provider.provider_id] = _load(
            ProviderSettings,
            prefix,
            environment,
            provider.defaults.model_dump(),
            host_paths=("home",),
        )
        secret_names.update(provider.secret_names)
    secrets = {
        name: SecretStr(environment[name].strip())
        for name in secret_names
        if environment.get(name, "").strip()
    }
    return Settings(
        agent=agent, container=container, providers=selections, secrets=secrets
    )


def environment_reference(providers: Sequence[ProviderEnvironment] = ()) -> str:
    """Generate the variable table from the same field definitions as the loader."""
    rows = ["| Variable | Meaning |", "| --- | --- |"]
    rows.append(
        "| `VIBESIM_AGENT_PROVIDERS_FILE` | Explicit named-provider YAML configuration |"
    )
    groups = [
        (AgentSettings, "VIBESIM_AGENT_", True),
        (ContainerSettings, "VIBESIM_RUNNER_", True),
    ]
    groups.extend(
        (ProviderSettings, profile.prefix, profile.accepts_home)
        for profile in providers
    )
    for model, prefix, accepts_home in groups:
        for name, field in model.model_fields.items():
            key = _environment_name(prefix, name, field)
            if key is None or (name == "home" and not accepts_home):
                continue
            rows.append(f"| `{key}` | {field.description} |")
    for name in sorted(
        {"OPENROUTER_API_KEY"} | {name for p in providers for name in p.secret_names}
    ):
        rows.append(
            f"| `{name}` | External credential; omitted from diagnostic output |"
        )
    return "\n".join(rows) + "\n"
