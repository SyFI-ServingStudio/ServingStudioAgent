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
    InvocationHome,
    RemoteInvocation,
    signal_remote,
)
from .mounts import WORKSPACE_TARGET

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
    stop_timeout: float

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
    stop_timeout: float = STOP_TIMEOUT

    @property
    def managed_context(self) -> str:
        return self.environment.managed_context

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
