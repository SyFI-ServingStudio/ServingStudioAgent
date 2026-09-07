"""Runtime configuration and stable path helpers for Docker-backed Codex calls."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from backend.agents_prompt import ensure_rendered

WORKSPACE = Path(__file__).resolve().parents[3]
UI_DIR = Path(__file__).resolve().parents[2]
MAIN_DIR = WORKSPACE / "main"
AGENT_WORKSPACES_ROOT = Path(
    os.environ.get("VIBESIM_WORKSPACES_ROOT", WORKSPACE / "agent-workspaces")
).expanduser()
# Legacy state is migration input only. New runtime code must not create data
# below user-facing-ui/workspaces.
LEGACY_WORKSPACES_DIR = UI_DIR / "workspaces"
PROMPTS_DIR = UI_DIR / "backend" / "prompts"
PROMPTS_CONTAINER_DIR = "/opt/vibesim/prompts"
# The mode-dependent prompts in PROMPTS_DIR are generated from
# `backend/prompt_templates/` and gitignored, so they may be absent (fresh
# clone) or stale (edited template) at this point. Rendering here — rather than
# in one entry point — is what makes that safe: every reader of PROMPTS_DIR
# imports this module, and there is no single startup path shared by `run.sh`,
# a bare `uvicorn`, and `unittest discover`. Idempotent; writes nothing when the
# tree is already current.
ensure_rendered()
ANALYZER_MCP_DIR = UI_DIR / "backend" / "analyzer_evidence_mcp"
ANALYZER_MCP_CONTAINER_DIR = "/opt/vibesim/analyzer-evidence-mcp"
ANALYZER_MCP_PYTHON = "/opt/vibesim-analyzer-mcp-venv/bin/python"
ANALYZER_MCP_SOURCE = os.environ.get("ANALYZER_MCP_SOURCE", "external").strip()
ANALYZER_MCP_BASE_URL = os.environ.get(
    "ANALYZER_MCP_BASE_URL",
    "http://host.docker.internal:8787",
).rstrip("/")

_default_image_owner = re.sub(
    r"[^a-z0-9_.-]+", "-", (os.environ.get("USER") or "codex").lower()
).strip("-.")
CODEX_DOCKER_IMAGE = os.environ.get(
    "CODEX_DOCKER_IMAGE",
    f"vibesim-ui-codex-runner:{_default_image_owner or 'codex'}",
)
CODEX_MODEL = os.environ.get("CODEX_MODEL", "gpt-5.6-sol")
# Codex reasoning effort, passed per call as `-c model_reasoning_effort=...`.
CODEX_REASONING_EFFORT = os.environ.get("CODEX_REASONING_EFFORT", "xhigh")
DEFAULT_CODEX_SERVICE_TIER = "default"
CODEXDS_MODEL = os.environ.get("CODEXDS_MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731")
CODEXDS_REASONING_EFFORT = os.environ.get("CODEXDS_REASONING_EFFORT", "max")
CLAUDE_MODEL_ALIASES = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-5"}
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
CLAUDE_MODEL = CLAUDE_MODEL_ALIASES.get(CLAUDE_MODEL, CLAUDE_MODEL)
# Explicit versions verified against this deployment. Aliases remain readable
# for existing conversations, but new selections pin the model version.
CLAUDE_MODELS = {
    "claude-sonnet-5": ("Claude Sonnet 5", ("low", "medium", "high", "xhigh", "max")),
    "claude-opus-5": ("Claude Opus 5", ("low", "medium", "high", "xhigh", "max")),
}
CLAUDE_AUTH_ENVIRONMENT = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
CLAUDE_ENVIRONMENT = (*CLAUDE_AUTH_ENVIRONMENT, "ANTHROPIC_BASE_URL")
# Bearer token gating the agent-facing HTTP API (/api/agent/*, /api/eval).
# Unset -> no auth, so local dev and the same-host eval harness keep working.
# Set it when exposing the backend to cross-machine agents.
VIBESIM_API_TOKEN = os.environ.get("VIBESIM_API_TOKEN", "").strip()
CODEX_IDLE_TIMEOUT = float(
    os.environ.get("CODEX_IDLE_TIMEOUT", os.environ.get("CODEX_TURN_TIMEOUT", "600"))
)
# Upstream statuses that mean the gateway, not the model, ended the call. The
# Codex CLI already reconnects about five times before reporting one, so a
# repair round would only spend another full reconnect cycle on the same outage.
# 4xx statuses outside this table stay ordinary warnings: a malformed or
# unauthorized request will not succeed on retry, so "try again" is wrong advice.
CODEX_TRANSPORT_FAILURE_STATUSES: dict[int, str] = {
    408: "upstream_unavailable",
    429: "upstream_rate_limited",
    500: "upstream_unavailable",
    502: "upstream_unavailable",
    503: "upstream_unavailable",
    504: "upstream_unavailable",
}
CODEX_DOCKER_GPUS = os.environ.get("CODEX_DOCKER_GPUS", "all").strip()
CODEX_DOCKER_UID = int(os.environ.get("CODEX_DOCKER_UID", str(os.getuid())))
CODEX_DOCKER_GID = int(os.environ.get("CODEX_DOCKER_GID", str(os.getgid())))
CODEX_DOCKER_USER = (
    os.environ.get("CODEX_DOCKER_USER") or os.environ.get("USER") or "codex"
)
CODEX_DOCKER_HOME = os.environ.get("CODEX_DOCKER_HOME", f"/home/{CODEX_DOCKER_USER}")
CODEX_DOCKER_CODEX_ROOT = f"{CODEX_DOCKER_HOME}/.vibesim-codex"
CODEX_DOCKER_DG_USE_LOCAL_VERSION = os.environ.get(
    "CODEX_DOCKER_DG_USE_LOCAL_VERSION", "0"
)
CODEX_DOCKER_UV_PROJECT_ENVIRONMENT = os.environ.get(
    "CODEX_DOCKER_UV_PROJECT_ENVIRONMENT",
    "/opt/vibesim-venv",
)
CODEX_DOCKER_UV_CACHE_DIR = os.environ.get(
    "CODEX_DOCKER_UV_CACHE_DIR", "/opt/vibesim-uv-cache"
)
_host_hf_home = os.environ.get("HF_HOME", "").strip()
HOST_HF_HOME = Path(_host_hf_home).expanduser() if _host_hf_home else None
CODEX_DOCKER_HF_HOME = "/model"
CONTAINER_RUNTIME_VERSION = os.environ.get(
    "CODEX_RUNNER_IMAGE_VERSION", "prebuilt-agent-runner-v11"
)
ORCHESTRATOR_SCHEMA_IN_CONTAINER = f"{PROMPTS_CONTAINER_DIR}/orchestrator.schema.json"
ASSISTANT_SCHEMA_IN_CONTAINER = f"{PROMPTS_CONTAINER_DIR}/assistant.schema.json"
# The implementer answers in an envelope too, because it now has two exits: back
# to the orchestrator, or straight to the user who interrupted it.
IMPLEMENTER_SCHEMA_IN_CONTAINER = f"{PROMPTS_CONTAINER_DIR}/implementer.schema.json"

EXECUTION_MODES = ("read-only", "workspace-write", "danger-full-access")
SANDBOX_MODES = EXECUTION_MODES
DEFAULT_SANDBOX = "workspace-write"

# How many Codex roles drive one turn. `orchestrated` keeps the historical
# orchestrator/implementer handoff loop; `single` runs one agent that both
# coordinates and implements, so it has no `delegate` action.
AGENT_MODES = ("orchestrated", "single")
DEFAULT_AGENT_MODE = "orchestrated"

# Roles that own a durable Codex home, session row, and container mount. Which
# subset a turn uses is decided by `roles_for_agent_mode`; the driving role
# (the one that emits the decision envelope) is the first entry.
AGENT_MODE_ROLES: dict[str, tuple[str, ...]] = {
    "orchestrated": ("orchestrator", "implementer"),
    "single": ("assistant",),
}
# Every role a conversation can store settings and a session for, in a stable
# order. A conversation row carries all of them so switching agent_mode before
# the first message does not lose the other mode's model selection.
ALL_CODEX_ROLES = ("orchestrator", "implementer", "assistant")
CODEX_ROLES = frozenset(ALL_CODEX_ROLES)

# The per-call driving-role prompt, and the schema constraining its output.
# Both are generated by `backend/agents_prompt.py`; do not hand-edit them.
AGENT_MODE_ROLE_PROMPT: dict[str, str] = {
    "orchestrated": "orchestrator.txt",
    "single": "assistant.txt",
}
AGENT_MODE_SCHEMA_FILE: dict[str, str] = {
    "orchestrated": "orchestrator.schema.json",
    "single": "assistant.schema.json",
}
AGENT_MODE_SCHEMA_IN_CONTAINER: dict[str, str] = {
    "orchestrated": ORCHESTRATOR_SCHEMA_IN_CONTAINER,
    "single": ASSISTANT_SCHEMA_IN_CONTAINER,
}

# (agent_mode, autonomous) -> the workspace contract bind-mounted at
# /workspace/AGENTS.md. Mirrors `backend/agents_prompt.AGENTS_PROMPT_MATRIX`;
# `tests/test_prompts.py` asserts the two stay in step.
AGENTS_PROMPT_MATRIX: dict[tuple[str, bool], str] = {
    ("orchestrated", False): "AGENTS.md",
    ("orchestrated", True): "AGENTS.autonomous.md",
    ("single", False): "AGENTS.single.md",
    ("single", True): "AGENTS.single.autonomous.md",
}


def normalize_agent_mode(agent_mode: str | None) -> str:
    return agent_mode if agent_mode in AGENT_MODES else DEFAULT_AGENT_MODE


def roles_for_agent_mode(agent_mode: str | None) -> tuple[str, ...]:
    """The Codex roles one turn of this mode runs, driving role first."""
    return AGENT_MODE_ROLES[normalize_agent_mode(agent_mode)]


def driving_role_for_agent_mode(agent_mode: str | None) -> str:
    """The role that emits the decision envelope and owns the turn's outcome."""
    return roles_for_agent_mode(agent_mode)[0]


LOG = logging.getLogger("vibesim_ui.codex_runtime")


@dataclass(frozen=True, slots=True)
class CodexFamilySpec:
    """One provider profile: an auth home, its models, and its env prerequisites.

    The family is the Codex session-compatibility boundary. Every model in a
    family shares the same ``CODEX_HOME``, so a rollout recorded by one model can
    be resumed by a sibling (verified against codex-cli 0.144.6: resuming a
    ``gpt-5.6-sol`` thread with ``gpt-5.6-luna`` succeeds and only emits a
    non-fatal advisory). Crossing families cannot resume: the session file lives
    in the other home and the provider and auth differ.
    """

    family_id: str
    label: str
    host_codex_home: Path
    catalog_filenames: tuple[str, ...]
    default_model: str
    default_effort: str
    fallback_efforts: tuple[str, ...]
    required_environment: tuple[str, ...] = ()
    runner: str = "codex"

    @property
    def available(self) -> bool:
        if self.runner == "claude":
            return any(
                os.environ.get(name, "").strip() for name in CLAUDE_AUTH_ENVIRONMENT
            )
        return self.host_codex_home.joinpath("config.toml").is_file() and all(
            os.environ.get(name, "").strip() for name in self.required_environment
        )


@dataclass(frozen=True, slots=True)
class CodexModelSpec:
    """One selectable model, resolved from its family's on-disk model catalog."""

    model_id: str
    label: str
    family_id: str
    efforts: tuple[str, ...]
    default_effort: str
    service_tiers: tuple[str, ...]

    @property
    def family(self) -> CodexFamilySpec:
        return CODEX_FAMILIES[self.family_id]

    @property
    def available(self) -> bool:
        return self.family.available


CODEX_FAMILIES: dict[str, CodexFamilySpec] = {
    "gpt": CodexFamilySpec(
        family_id="gpt",
        label="GPT-5.6",
        host_codex_home=Path(
            os.environ.get("CODEX_TRADITIONAL_HOME", Path.home() / ".codex")
        ).expanduser(),
        catalog_filenames=("models_cache.json", "models_catalog.json"),
        default_model=CODEX_MODEL,
        default_effort=CODEX_REASONING_EFFORT,
        fallback_efforts=("low", "medium", "high", "xhigh"),
    ),
    "deepseek": CodexFamilySpec(
        family_id="deepseek",
        label="DeepSeek",
        host_codex_home=Path(
            os.environ.get("CODEXDS_HOME", Path.home() / ".codex-ds")
        ).expanduser(),
        catalog_filenames=("models_catalog.json", "models_cache.json"),
        default_model=CODEXDS_MODEL,
        default_effort=CODEXDS_REASONING_EFFORT,
        fallback_efforts=("high", "xhigh", "max"),
        required_environment=("VLLM_API_KEY",),
    ),
    "claude": CodexFamilySpec(
        family_id="claude",
        label="Claude",
        host_codex_home=Path.home() / ".claude",
        catalog_filenames=(),
        default_model=CLAUDE_MODEL,
        default_effort="high",
        fallback_efforts=("low", "medium", "high"),
        runner="claude",
    ),
}

# Which catalog entries a conversation may actually pick. The catalogs carry far
# more (older generations, hidden internal models); this keeps the selector to
# the models this deployment is meant to run.
MODEL_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "claude": tuple(dict.fromkeys((CLAUDE_MODEL, *CLAUDE_MODELS))),
    "gpt": ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"),
    "deepseek": (CODEXDS_MODEL,),
}

DEFAULT_CODEX_FAMILY = "gpt"
DEFAULT_CODEX_MODEL = CODEX_FAMILIES[DEFAULT_CODEX_FAMILY].default_model
DEFAULT_CODEX_EFFORT = CODEX_FAMILIES[DEFAULT_CODEX_FAMILY].default_effort

# Pre-registry conversations stored a backend id. Reads map them onto models.
LEGACY_BACKEND_MODELS: dict[str, str] = {
    **CLAUDE_MODEL_ALIASES,
    "traditional": CODEX_FAMILIES["gpt"].default_model,
    "codexds": CODEX_FAMILIES["deepseek"].default_model,
}

_MODEL_REGISTRY_CACHE: dict[
    str, tuple[tuple[float, ...], dict[str, CodexModelSpec]]
] = {}


def _catalog_models(family: CodexFamilySpec) -> dict[str, dict]:
    """Read one family's on-disk catalog, keyed by model slug.

    Both catalog formats (`models_cache.json` written by the OpenAI client and
    the hand-authored `models_catalog.json` used by the vLLM profile) share the
    same ``models: [{slug, display_name, supported_reasoning_levels, ...}]``
    shape, so a single reader covers both.
    """
    for filename in family.catalog_filenames:
        path = family.host_codex_home / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            LOG.warning("unreadable Codex model catalog: %s", path)
            continue
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            continue
        return {
            str(entry["slug"]): entry
            for entry in models
            if isinstance(entry, dict) and entry.get("slug")
        }
    return {}


def _family_registry(family: CodexFamilySpec) -> dict[str, CodexModelSpec]:
    catalog = _catalog_models(family)
    registry: dict[str, CodexModelSpec] = {}
    for model_id in MODEL_ALLOWLIST.get(family.family_id, ()):
        entry = catalog.get(model_id)
        efforts = tuple(
            str(level["effort"])
            for level in (entry or {}).get("supported_reasoning_levels", [])
            if isinstance(level, dict) and level.get("effort")
        )
        # A missing or stale catalog must not remove a configured model from the
        # UI; fall back to the family's conservative effort ladder instead.
        if family.runner == "claude" and model_id in CLAUDE_MODELS:
            efforts = CLAUDE_MODELS[model_id][1]
        efforts = efforts or family.fallback_efforts
        default_effort = (
            family.default_effort
            if family.default_effort in efforts
            else str((entry or {}).get("default_reasoning_level") or efforts[-1])
        )
        additional_service_tiers = tuple(
            str(service_tier)
            for service_tier in (entry or {}).get("additional_speed_tiers", [])
            if service_tier
        )
        service_tiers = tuple(
            dict.fromkeys((DEFAULT_CODEX_SERVICE_TIER, *additional_service_tiers))
        )
        registry[model_id] = CodexModelSpec(
            model_id=model_id,
            label=str(
                (entry or {}).get("display_name")
                or (
                    CLAUDE_MODELS.get(model_id, (model_id, ()))[0]
                    if family.runner == "claude"
                    else model_id
                )
            ),
            family_id=family.family_id,
            efforts=efforts,
            default_effort=(
                default_effort if default_effort in efforts else efforts[-1]
            ),
            service_tiers=service_tiers,
        )
    return registry


def codex_model_registry() -> dict[str, CodexModelSpec]:
    """All selectable models, refreshed when a family's catalog file changes."""
    registry: dict[str, CodexModelSpec] = {}
    for family in CODEX_FAMILIES.values():
        stamp = tuple(
            family.host_codex_home.joinpath(name).stat().st_mtime
            if family.host_codex_home.joinpath(name).is_file()
            else 0.0
            for name in family.catalog_filenames
        )
        cached = _MODEL_REGISTRY_CACHE.get(family.family_id)
        if cached is None or cached[0] != stamp:
            cached = (stamp, _family_registry(family))
            _MODEL_REGISTRY_CACHE[family.family_id] = cached
        registry.update(cached[1])
    return registry


def codex_family(family_id: str) -> CodexFamilySpec:
    try:
        return CODEX_FAMILIES[family_id]
    except KeyError as exc:
        raise ValueError(f"unsupported Codex family: {family_id!r}") from exc


def codex_model(model_id: str) -> CodexModelSpec:
    resolved = LEGACY_BACKEND_MODELS.get(model_id, model_id)
    try:
        return codex_model_registry()[resolved]
    except KeyError as exc:
        raise ValueError(f"unsupported Codex model: {model_id!r}") from exc


def normalize_role_runtime(
    model_id: str | None,
    effort: str | None,
    service_tier: str | None = None,
) -> dict[str, str]:
    """Coerce one role's stored or requested selection onto the live registry."""
    try:
        model = codex_model(model_id or DEFAULT_CODEX_MODEL)
    except ValueError:
        model = codex_model(DEFAULT_CODEX_MODEL)
    chosen_effort = effort if effort in model.efforts else model.default_effort
    chosen_service_tier = (
        service_tier
        if service_tier in model.service_tiers
        else DEFAULT_CODEX_SERVICE_TIER
    )
    return {
        "model": model.model_id,
        "effort": chosen_effort,
        "service_tier": chosen_service_tier,
    }


def codex_model_catalog() -> list[dict[str, object]]:
    return [
        {
            "id": model.model_id,
            "label": model.label,
            "family": model.family_id,
            "familyLabel": model.family.label,
            "runner": model.family.runner,
            "efforts": list(model.efforts),
            "defaultEffort": model.default_effort,
            "serviceTiers": list(model.service_tiers),
            "defaultServiceTier": DEFAULT_CODEX_SERVICE_TIER,
            "available": model.available,
        }
        for model in codex_model_registry().values()
    ]


def codex_family_catalog() -> list[dict[str, object]]:
    return [
        {
            "id": family.family_id,
            "label": family.label,
            "runner": family.runner,
            "available": family.available,
            "requiredEnvironment": list(family.required_environment),
            "credentialEnvironmentAlternatives": (
                list(CLAUDE_AUTH_ENVIRONMENT) if family.runner == "claude" else []
            ),
        }
        for family in CODEX_FAMILIES.values()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MAIN_LOCK_SHA = os.environ.get("CODEX_MAIN_LOCK_SHA") or _sha256_file(
    MAIN_DIR / "uv.lock"
)


def agents_prompt_name(
    autonomous: bool,
    agent_mode: str = DEFAULT_AGENT_MODE,
) -> str:
    return AGENTS_PROMPT_MATRIX[(normalize_agent_mode(agent_mode), bool(autonomous))]


def prompt_files_for(autonomous: bool, agent_mode: str) -> tuple[str, ...]:
    """Every prompt artifact that shapes one turn, for hashing and diagnostics."""
    mode = normalize_agent_mode(agent_mode)
    names = [
        agents_prompt_name(autonomous, mode),
        AGENT_MODE_ROLE_PROMPT[mode],
        AGENT_MODE_SCHEMA_FILE[mode],
    ]
    if "implementer" in AGENT_MODE_ROLES[mode]:
        names.insert(2, "implementer.txt")
    return tuple(names)


def prompt_fingerprint(
    *,
    autonomous: bool = False,
    agent_mode: str = DEFAULT_AGENT_MODE,
    role_runtimes: dict[str, dict[str, str]] | None = None,
) -> str:
    """Hash the effective role contract for diagnostics and provenance.

    Reasoning effort is part of the hash: it is a per-call Codex option, so two
    turns of one conversation can legitimately carry different fingerprints.
    Only the roles the mode actually runs contribute, so an unused role's stored
    model selection cannot change the fingerprint.
    """
    mode = normalize_agent_mode(agent_mode)
    digest = hashlib.sha256()
    for name in prompt_files_for(autonomous, mode):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((PROMPTS_DIR / name).read_bytes())
        digest.update(b"\0")
    for role in roles_for_agent_mode(mode):
        runtime = (role_runtimes or {}).get(role)
        selection = normalize_role_runtime(
            (runtime or {}).get("model"),
            (runtime or {}).get("effort"),
            (runtime or {}).get("service_tier"),
        )
        model = codex_model(selection["model"])
        digest.update(role.encode("utf-8"))
        digest.update(b"\0")
        digest.update(model.family_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(model.model_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(selection["effort"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(selection["service_tier"].encode("utf-8"))
    digest.update(b"\0autonomous=")
    digest.update(str(autonomous).encode("utf-8"))
    digest.update(b"\0agent_mode=")
    digest.update(mode.encode("utf-8"))
    return digest.hexdigest()[:16]


def workspace_main_for(workspace_id: str) -> Path:
    """Return the conventional repo path for one workspace.

    Descriptors are authoritative at the API boundary. Runtime workspace
    creation uses the same fixed layout, so lower-level Docker helpers do not
    need to parse mutable JSON on every call.
    """
    if workspace_id == "w_main":
        return MAIN_DIR
    return AGENT_WORKSPACES_ROOT / workspace_id / "repo"


def codex_home_for(workspace_id: str, conversation_id: str) -> Path:
    return AGENT_WORKSPACES_ROOT / workspace_id / "codex" / conversation_id


def role_codex_home_for(workspace_id: str, conversation_id: str, role: str) -> Path:
    if role not in CODEX_ROLES:
        raise ValueError(f"unsupported Codex role: {role!r}")
    return codex_home_for(workspace_id, conversation_id) / role


def role_codex_home_in_container(role: str) -> str:
    if role not in CODEX_ROLES:
        raise ValueError(f"unsupported Codex role: {role!r}")
    return f"{CODEX_DOCKER_CODEX_ROOT}/{role}"


def container_name(workspace_id: str, conversation_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{workspace_id}-{conversation_id}")[:48]
    return f"vibesim-ui-{safe}"
