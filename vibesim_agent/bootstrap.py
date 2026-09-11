"""Application entry points; configuration and state access happen only on call."""

import hashlib
import os
import pwd
from collections.abc import Mapping
from pathlib import Path

from fastapi import FastAPI

from .application import build_application
from .composition import build_builtin_setup
from .providers.builtin import provider_environments
from .runtime.workspace import WorkspaceSnapshot
from .settings import ConfigurationError, Settings, load_settings
from .storage.database import Database
from .storage.ownership import WorkspaceOwnership
from .storage.registry import WorkspaceRegistry

# These are retired Agent configuration names, not arbitrary Codex CLI settings.
RETIRED_ENVIRONMENT = frozenset(
    {
        "VIBESIM_WORKSPACES_ROOT",
        "VIBESIM_MAIN_DIR",
        "VIBESIM_API_TOKEN",
        "VIBESIM_MANAGED_BACKEND_URL",
        "VIBESIM_NAMING_MODEL",
        "VIBESIM_NAMING_BASE_URL",
        "VIBESIM_NAMING_TIMEOUT_SECONDS",
        "OPENROUTE_KEY",
        "CODEX_MODEL",
        "CODEX_REASONING_EFFORT",
        "CODEX_TRADITIONAL_HOME",
        "CODEXDS_MODEL",
        "CODEXDS_REASONING_EFFORT",
        "CODEXDS_HOME",
        "CLAUDE_MODEL",
        "CODEX_IDLE_TIMEOUT",
        "CODEX_TURN_TIMEOUT",
        "CODEX_MAIN_LOCK_SHA",
        "CODEX_DOCKER_IMAGE",
        "CODEX_DOCKER_UID",
        "CODEX_DOCKER_GID",
        "CODEX_DOCKER_USER",
        "CODEX_DOCKER_HOME",
        "CODEX_DOCKER_GPUS",
        "CODEX_DOCKER_DG_USE_LOCAL_VERSION",
        "CODEX_DOCKER_UV_PROJECT_ENVIRONMENT",
        "CODEX_DOCKER_UV_CACHE_DIR",
        "CODEX_RUNNER_IMAGE_VERSION",
        "ANALYZER_MCP_SOURCE",
        "ANALYZER_MCP_BASE_URL",
    }
)
MANAGED_CONTEXT = "/opt/vibesim/managed/context.json"


def host_home(environment: Mapping[str, str]) -> Path:
    home = Path(environment.get("HOME") or pwd.getpwuid(os.getuid()).pw_dir)
    if not home.is_absolute():
        raise ConfigurationError("Invalid configuration: HOME")
    return home


def configuration(
    *, environment: Mapping[str, str] | None = None, repo_root: Path | None = None
) -> Settings:
    environment = dict(os.environ if environment is None else environment)
    retired = sorted(RETIRED_ENVIRONMENT.intersection(environment))
    if retired:
        raise ConfigurationError(
            "Retired Agent configuration keys: " + ", ".join(retired)
        )
    return load_settings(
        environment=environment,
        repo_root=repo_root,
        providers=provider_environments(host_home(environment)),
    )


def _source(settings: Settings, environment: Mapping[str, str]) -> tuple[Path, ...]:
    main = settings.agent.main_dir.resolve()
    for name in ("AGENTS.md", "uv.lock"):
        if not (main / name).is_file():
            raise ConfigurationError(f"VIBESIM_AGENT_MAIN_DIR requires {name}")
    if (main / "AGENTS.md").is_symlink():
        raise ConfigurationError("VIBESIM_AGENT_MAIN_DIR requires a regular AGENTS.md")
    _, gitlinks = WorkspaceSnapshot(
        main, process_environment=environment
    ).tracked_entries()
    submodules = tuple(path for path in gitlinks if (main / path).is_dir())
    if any(not (main / path).resolve().is_relative_to(main) for path in submodules):
        raise ConfigurationError("VIBESIM_AGENT_MAIN_DIR has an external gitlink path")
    return submodules


def initialize_state(
    *, environment: Mapping[str, str] | None = None, repo_root: Path | None = None
) -> dict:
    environment = dict(os.environ if environment is None else environment)
    settings = configuration(environment=environment, repo_root=repo_root)
    _source(settings, environment)
    return WorkspaceRegistry(settings.agent.workspaces_root).initialize_main(
        settings.agent.main_dir,
        base_revision=WorkspaceSnapshot(
            settings.agent.main_dir, process_environment=environment
        ).revision(),
    )


def create_application(
    *, environment: Mapping[str, str] | None = None, repo_root: Path | None = None
) -> FastAPI:
    """Uvicorn factory for existing initialized or offline-migrated state."""
    environment = dict(os.environ if environment is None else environment)
    settings = configuration(environment=environment, repo_root=repo_root)
    submodules = _source(settings, environment)
    registry = WorkspaceRegistry(settings.agent.workspaces_root)
    try:
        registry.get("w_main")
    except KeyError:
        raise ConfigurationError(
            "Workspace state requires explicit init or offline migration before serving"
        ) from None
    for workspace in registry.list(include_archived=True):
        with Database(registry.database_path(workspace["workspace_id"])).connect():
            pass
    mcp = settings.agent.repo_root / "vibesim_agent" / "analyzer_evidence_mcp"
    if not (mcp / "server.py").is_file():
        raise ConfigurationError("Agent checkout is missing the Analyzer MCP server")

    def providers(home, execution_environment, prompts):
        return build_builtin_setup(
            settings, home, execution_environment, prompts, host_environment=environment
        )

    namespace = "vibesim-" + hashlib.sha256(str(registry.root).encode()).hexdigest()
    ownership = WorkspaceOwnership(registry.root)
    ownership.acquire()
    try:
        prompts_directory = registry.root / ".prompts"
        if prompts_directory.is_symlink() or (
            prompts_directory.exists()
            and (
                not prompts_directory.is_dir()
                or any(path.is_symlink() for path in prompts_directory.iterdir())
            )
        ):
            raise ConfigurationError(
                "workspace prompts must not contain symbolic links"
            )
        app = build_application(
            settings,
            providers=providers,
            prompts_directory=prompts_directory,
            mcp_directory=mcp,
            managed_context=MANAGED_CONTEXT,
            namespace=namespace,
            submodules=submodules,
            workspace_environment=environment,
            ownership=ownership,
        )
    except BaseException:
        ownership.close()
        raise
    app.state.settings = settings
    return app
