"""Where a turn runs, as a per-turn value the adapters can stay ignorant of."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from .command import ExecutionEnvironment
from .invocation import (
    SIGNAL_GRACE,
    SIGNAL_TIMEOUT,
    STOP_TIMEOUT,
    HostInvocation,
    InvocationHome,
    RemoteInvocation,
    signal_remote,
)
from .mounts import WORKSPACE_TARGET
from .process import KillProcess, kill_process, kill_process_group

# The pid file is written by the process itself rather than read from the
# spawner, because under `docker exec` the local client and the remote shell are
# different processes: the client can exit before the shell has started, and the
# `.cancel` check closes that window.
PID_WRAPPER = (
    'printf "%s\\n" "$$" > "$1" || exit; '
    'if [ -e "$1.cancel" ]; then rm -f -- "$1"; exit 130; fi; '
    'shift; exec "$@"'
)


class Invocation(Protocol):
    """One call's cancellation surface, owned by whatever transport spawned it."""

    pid_file: str

    def mark_cancelled(self) -> None: ...

    async def stop(self, process: asyncio.subprocess.Process) -> None: ...

    def remove_pid(self) -> None: ...


class Execution(Protocol):
    """Everything that differs between running in a container and on the host.

    Carried per turn rather than resolved per adapter: five of these values are
    turn-scoped (the container name, the role home path, the cancellation
    transport), while the one process-global object available to an adapter is a
    frozen singleton shared by every conversation.
    """

    cwd: Path | None
    agent_prompt: str
    managed_context: str
    mcp_python: str
    mcp_server: str
    analyzer_source: str
    analyzer_base_url: str
    stop_timeout: float
    start_new_session: bool
    kill: KillProcess

    def command(
        self,
        arguments: list[str],
        *,
        environment: Mapping[str, str],
        inherited: tuple[str, ...],
        pid_file: str,
        label: str,
    ) -> list[str]: ...

    def spawn_environment(
        self, process_environment: Mapping[str, str] | None
    ) -> dict[str, str] | None: ...

    def home_path(self, home: InvocationHome) -> str: ...

    def invocation(
        self,
        *,
        execution_id: str,
        home: InvocationHome,
        logger: logging.Logger,
        label: str,
        process_environment: Mapping[str, str] | None,
    ) -> Invocation: ...


@dataclass(frozen=True)
class DockerExecution:
    """Today's transport, unchanged: `docker exec` into a per-conversation runner."""

    environment: ExecutionEnvironment
    container: str

    cwd: Path | None = None
    agent_prompt: str = str(PurePosixPath(WORKSPACE_TARGET) / "AGENTS.md")
    # Baked into the runner image; on the host neither path exists, so the
    # interpreter has to come from the turn rather than from the command builder.
    mcp_python: str = "/opt/vibesim-analyzer-mcp-venv/bin/python"
    mcp_server: str = "/opt/vibesim/analyzer-evidence-mcp/server.py"
    stop_timeout: float = STOP_TIMEOUT
    # The CLI leads no group of its own: it is a `docker exec` client, and the
    # process that matters lives in the container where killpg cannot reach.
    start_new_session: bool = False
    kill: KillProcess = staticmethod(kill_process)

    @property
    def managed_context(self) -> str:
        return self.environment.managed_context

    @property
    def analyzer_source(self) -> str:
        return self.environment.agent.analyzer_source

    @property
    def analyzer_base_url(self) -> str:
        # `host.docker.internal` by default, which only resolves inside a
        # container; host turns need the same address under another name.
        return self.environment.agent.analyzer_base_url

    def prefix(
        self,
        *,
        environment: Mapping[str, str],
        inherited: tuple[str, ...] = (),
    ) -> list[str]:
        return self.environment.prefix(
            self.container, environment=environment, inherited=inherited
        )

    def command(
        self,
        arguments: list[str],
        *,
        environment: Mapping[str, str],
        inherited: tuple[str, ...] = (),
        pid_file: str,
        label: str,
    ) -> list[str]:
        return [
            *self.prefix(environment=environment, inherited=inherited),
            "sh",
            "-c",
            PID_WRAPPER,
            label,
            pid_file,
            *arguments,
        ]

    def spawn_environment(
        self, process_environment: Mapping[str, str] | None
    ) -> dict[str, str] | None:
        # The Docker client's own environment, credentials included: `-e NAME`
        # inherits by name from this process.
        return dict(process_environment) if process_environment is not None else None

    def home_path(self, home: InvocationHome) -> str:
        return home.container

    def invocation(
        self,
        *,
        execution_id: str,
        home: InvocationHome,
        logger: logging.Logger,
        label: str,
        process_environment: Mapping[str, str] | None,
        signal_grace: float | None = None,
    ) -> RemoteInvocation:
        # Read at call time, not bound as defaults: the signal budget is what
        # tests shorten, and a default argument would freeze it at import.
        async def signal(container: str, pid_file: str, name: str) -> None:
            await signal_remote(
                container,
                pid_file,
                name,
                timeout=SIGNAL_TIMEOUT,
                label=label,
                environment=process_environment,
            )

        return RemoteInvocation(
            execution_id=execution_id,
            container=self.container,
            home=home,
            signal=signal,
            logger=logger,
            label=label,
            signal_grace=SIGNAL_GRACE if signal_grace is None else signal_grace,
        )


# Container-only names. Every one of these points at a path or a device that
# does not exist on the host, and `uv run` in a worktree fails outright when it
# inherits UV_PROJECT_ENVIRONMENT or UV_CACHE_DIR from the runner image.
CONTAINER_ONLY_VARIABLES = frozenset(
    {
        "HOME",
        "UV_PROJECT_ENVIRONMENT",
        "UV_CACHE_DIR",
        "VIBESIM_EXPECTED_LOCK_SHA",
        "VIBESIM_RUNNER_GPUS",
        "NVIDIA_DRIVER_CAPABILITIES",
        "DG_USE_LOCAL_VERSION",
    }
)


@dataclass(frozen=True)
class HostExecution:
    """Run the CLI here, in a real worktree, with no container in between."""

    repo: Path
    agent_prompt: str
    managed_context: str
    analyzer_source: str
    analyzer_base_url: str
    mcp_python: str
    mcp_server: str

    stop_timeout: float = STOP_TIMEOUT
    # The CLI is the process, so it leads its own group and the group is what
    # has to be signalled: Codex and Claude both start children of their own.
    start_new_session: bool = True
    kill: KillProcess = staticmethod(kill_process_group)

    def __post_init__(self):
        if not self.repo.is_absolute():
            raise ValueError("host execution requires an absolute repository path")

    @property
    def cwd(self) -> Path:
        return self.repo

    def command(
        self,
        arguments: list[str],
        *,
        environment: Mapping[str, str],
        inherited: tuple[str, ...] = (),
        pid_file: str,
        label: str,
    ) -> list[str]:
        del pid_file, label
        # No pid wrapper and no shell. The wrapper solves a `docker exec`
        # problem that does not exist here, and threading argv that carries
        # whole JSON schemas through shell quoting would be a new injection
        # surface for nothing. `env` is not a shell: it assigns and execs, so
        # nothing in these values is ever parsed.
        variables = self._variables(environment)
        if set(inherited).intersection(variables):
            raise ValueError(
                "inherited environment conflicts with explicit runtime values"
            )
        # The host counterpart of `docker exec -e`. Without it CODEX_HOME and
        # CLAUDE_CONFIG_DIR never reach the CLI, which would then quietly use
        # the operator's own `~/.codex` instead of the role home.
        return [
            "env",
            *(f"{name}={value}" for name, value in variables.items()),
            *arguments,
        ]

    def _variables(self, environment: Mapping[str, str]) -> dict[str, str]:
        return {
            **environment,
            "ANALYZER_MCP_SOURCE": self.analyzer_source,
            "ANALYZER_MCP_BASE_URL": self.analyzer_base_url,
            "VIBESIM_MANAGED_RUN_CONTEXT": self.managed_context,
            "VIBESIM_MANAGED_JOB_CONTEXT": self.managed_context,
        }

    def spawn_environment(
        self, process_environment: Mapping[str, str] | None
    ) -> dict[str, str] | None:
        if process_environment is None:
            return None
        return {
            key: value
            for key, value in process_environment.items()
            if key not in CONTAINER_ONLY_VARIABLES
        }

    def home_path(self, home: InvocationHome) -> str:
        return str(home.host)

    def invocation(
        self,
        *,
        execution_id: str,
        home: InvocationHome,
        logger: logging.Logger,
        label: str,
        process_environment: Mapping[str, str] | None = None,
        signal_grace: float | None = None,
    ) -> HostInvocation:
        del process_environment
        return HostInvocation(
            execution_id=execution_id,
            home=home,
            logger=logger,
            label=label,
            signal_grace=SIGNAL_GRACE if signal_grace is None else signal_grace,
        )
