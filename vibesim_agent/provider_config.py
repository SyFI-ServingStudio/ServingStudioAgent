"""Strict named-provider YAML: public connection settings and credential references."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import SecretStr, ValidationError

from .settings import (
    ConfigurationError,
    ConnectionModelSettings,
    ConnectionSettings,
    ProviderSettings,
    validate_provider_id,
)

_ROLES = {"orchestrator", "implementer", "assistant"}
_CLAUDE_AUTH = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"}
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
_CODEX_CREDENTIAL = re.compile(r"[A-Z][A-Z0-9_]*(?:_KEY|_TOKEN|_SECRET)")
_CONTROL_NAMES = {"PATH", "HOME", "USER", "SHELL", "TMPDIR"}
_CONTROL_PREFIXES = ("PYTHON", "LD_", "DOCKER", "UV_")
_FIELDS = {
    "adapter",
    "label",
    "environment",
    "base_url",
    "session_identity",
    "home",
    "default_model",
    "models",
    "default_effort",
    "service_tier",
}


@dataclass(frozen=True)
class ProviderConfiguration:
    providers: dict[str, ProviderSettings]
    connections: dict[str, ConnectionSettings]
    role_providers: dict[str, str]
    secrets: dict[str, SecretStr] = field(repr=False)


def _invalid():
    raise ConfigurationError("Invalid providers file configuration") from None


def _text(value):
    if not isinstance(value, str) or not value.strip():
        _invalid()
    return value


def _mapping(value):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _invalid()
    return value


def _reference(value):
    if not isinstance(value, str) or _ENV_NAME.fullmatch(value) is None:
        _invalid()
    if value in _CONTROL_NAMES or value.startswith(_CONTROL_PREFIXES):
        _invalid()
    return value


def _read(path):
    import yaml

    class StrictLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str) or key in result:
                    _invalid()
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    if not path.is_absolute():
        _invalid()
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        details = os.fstat(fd)
        if not stat.S_ISREG(details.st_mode) or details.st_size > 1024 * 1024:
            _invalid()
        with os.fdopen(fd, "rb") as stream:
            fd = None
            content = stream.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            _invalid()
        # Aliases/anchors are unnecessary for this small schema and can create recursive data.
        for event in yaml.parse(content):
            if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None):
                _invalid()
        return yaml.load(content, Loader=StrictLoader)
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError):
        _invalid()
    finally:
        if fd is not None:
            os.close(fd)


def load_provider_config(
    path: Path, *, environment: Mapping[str, str]
) -> ProviderConfiguration:
    """Load once, without modifying the file or consulting ambient os.environ."""
    document = _mapping(_read(path))
    if (
        set(document) != {"version", "providers", "defaults"}
        or type(document["version"]) is not int
        or document["version"] != 1
    ):
        _invalid()
    declarations = _mapping(document["providers"])
    defaults = _mapping(document["defaults"])
    if not declarations or set(defaults) != _ROLES:
        _invalid()
    selections, connections, secrets = {}, {}, {}
    for provider_id, raw in declarations.items():
        try:
            validate_provider_id(provider_id)
        except ConfigurationError:
            _invalid()
        value = _mapping(raw)
        if set(value) - _FIELDS or not {
            "adapter",
            "default_model",
            "default_effort",
            "models",
        } <= set(value):
            _invalid()
        adapter = value["adapter"]
        if not isinstance(adapter, str) or adapter not in {"codex", "claude"}:
            _invalid()
        refs = dict(_mapping(value.get("environment", {})))
        if adapter == "claude" and (set(refs) - _CLAUDE_AUTH or len(refs) > 1):
            _invalid()
        if adapter == "claude" and bool(refs) == ("home" in value):
            _invalid()
        for target, source in list(refs.items()):
            if adapter == "codex" and _CODEX_CREDENTIAL.fullmatch(target) is None:
                _invalid()
            _reference(target)
            if isinstance(source, dict):
                if set(source) != {"value"}:
                    _invalid()
                credential = _text(source["value"]).strip()
                source = f"inline:{provider_id}:{target}"
                refs[target] = source
            else:
                _reference(source)
                credential = environment.get(source, "").strip()
            if credential:
                secrets[source] = SecretStr(credential)
        endpoint = value.get("base_url")
        if endpoint is not None:
            if adapter != "claude":
                _invalid()
            endpoint = _text(endpoint)
            try:
                parsed = urlsplit(endpoint)
                if (
                    parsed.scheme not in {"http", "https"}
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                    or parsed.query
                    or parsed.fragment
                ):
                    _invalid()
                if parsed.port is not None and not 1 <= parsed.port <= 65535:
                    _invalid()
            except ValueError:
                _invalid()
            endpoint = endpoint.rstrip("/")
        optional = {}
        for key in ("label", "session_identity"):
            if key in value:
                optional[key] = _text(value[key])
        models = _mapping(value["models"])
        if not models:
            _invalid()
        declarations = []
        for model_id, raw_model in models.items():
            model_id = _text(model_id)
            raw_model = _mapping(raw_model)
            if set(raw_model) != {"efforts"}:
                _invalid()
            efforts = raw_model["efforts"]
            if not isinstance(efforts, list) or not efforts:
                _invalid()
            efforts = tuple(_text(item) for item in efforts)
            if len(set(efforts)) != len(efforts):
                _invalid()
            declarations.append(
                ConnectionModelSettings(model_id=model_id, efforts=efforts)
            )
        profile = {
            "model": _text(value["default_model"]),
            "effort": _text(value["default_effort"]),
            **(
                {"service_tier": _text(value["service_tier"])}
                if "service_tier" in value
                else {}
            ),
        }
        model_ids = [model.model_id for model in declarations]
        if profile["model"] not in model_ids:
            _invalid()
        if any(profile["effort"] not in model.efforts for model in declarations):
            _invalid()
        optional["models"] = tuple(declarations)
        if "home" in value:
            home = Path(_text(value["home"]))
            if home.parts and home.parts[0] == "~":
                home = Path(_text(environment.get("HOME"))).joinpath(*home.parts[1:])
            profile["home"] = home
        try:
            selections[provider_id] = ProviderSettings.model_validate(profile)
            connections[provider_id] = ConnectionSettings(
                adapter=adapter, environment=refs, base_url=endpoint, **optional
            )
        except ValidationError:
            _invalid()
    if any(
        not isinstance(value, str) or value not in selections
        for value in defaults.values()
    ):
        _invalid()
    return ProviderConfiguration(selections, connections, dict(defaults), secrets)
