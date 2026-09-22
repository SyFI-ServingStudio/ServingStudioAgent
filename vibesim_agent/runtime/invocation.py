"""Stop one container invocation without losing delayed startup cancellation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .host import PGID_SUFFIX, record_process_group

SIGNAL_TIMEOUT = 5
SIGNAL_GRACE = 5
# Include helper timeout/reap and remote grace for all three signal attempts.
STOP_TIMEOUT = 3 * (2 * SIGNAL_TIMEOUT + SIGNAL_GRACE) + 5


@dataclass(frozen=True)
class InvocationHome:
    host: Path
    container: str


@dataclass(frozen=True)
class RoleContext:
    """What a role home needs that depends on where the turn will run.

    `skills` is a symlink target: `/workspace/skills` under a mount, the
    worktree's own directory on the host. `global_prompt` is set only on the
    host, where Codex reads `$CODEX_HOME/AGENTS.md` -- in a container the role
    contract arrives as a bind mount over the workspace's own AGENTS.md instead.
    """

    skills: str
    global_prompt: Path | None = None


async def signal_remote(
    container: str,
    pid_file: str,
    signal: str,
    *,
    timeout: float = SIGNAL_TIMEOUT,
    label: str,
    environment: Mapping[str, str] | None = None,
) -> None:
    process = await asyncio.create_subprocess_exec(
        "docker",
        "exec",
        container,
        "sh",
        "-c",
        'if [ -f "$1" ]; then kill -"$2" "$(cat "$1")" 2>/dev/null || true; fi',
        "vibesim-signal",
        pid_file,
        signal,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=environment,
    )
    try:
        await asyncio.wait_for(process.wait(), timeout)
        if process.returncode != 0:
            raise RuntimeError(f"{label} signal helper exited {process.returncode}")
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await asyncio.wait_for(process.wait(), timeout)


class RemoteInvocation:
    def __init__(
        self,
        *,
        execution_id: str,
        container: str,
        home: InvocationHome,
        signal: Callable[[str, str, str], Awaitable[None]],
        logger: logging.Logger,
        label: str,
        signal_grace: float = SIGNAL_GRACE,
    ):
        self.container = container
        self.signal = signal
        self.logger = logger
        self.label = label
        self.signal_grace = signal_grace
        pid_name = f"call-{execution_id}.pid"
        self.pid_file = str(PurePosixPath(home.container) / pid_name)
        self.host_pid = home.host / pid_name
        self.cancel_marker = home.host / f"{pid_name}.cancel"
        self._marker_attempted = False
        self._marker_error: OSError | None = None

    def record(self, process: asyncio.subprocess.Process) -> None:
        """Nothing to note: the container is the reaping unit, not the group.

        The local process here is a `docker exec` client. Whatever it leaves
        running lives in the container and goes away with it.
        """

    def mark_cancelled(self) -> None:
        if self._marker_attempted:
            return
        self._marker_attempted = True
        try:
            self.cancel_marker.touch(exist_ok=True)
        except OSError as error:
            self._marker_error = error
            self.logger.error(
                "Could not persist %s startup cancellation: %s", self.label, error
            )

    async def stop(self, process: asyncio.subprocess.Process) -> None:
        # Keep the marker until role-home deletion: the remote shell can start
        # after its local Docker client exits. Each invocation needs a unique ID.
        self.mark_cancelled()
        if not self.host_pid.exists():
            if self._marker_error is not None:
                raise RuntimeError(
                    f"Cannot prevent delayed {self.label} startup: cancellation marker failed"
                ) from self._marker_error
            return
        for signal in ("INT", "TERM", "KILL"):
            try:
                await self.signal(self.container, self.pid_file, signal)
            except Exception as error:  # noqa: BLE001 - later signals must still run
                self.logger.warning(
                    "%s %s signal failed: %s", self.label, signal, error
                )
            try:
                await asyncio.wait_for(process.wait(), self.signal_grace)
                return
            except TimeoutError:
                pass
        raise RuntimeError(f"{self.label} invocation did not stop after INT/TERM/KILL")

    def remove_pid(self) -> None:
        with contextlib.suppress(OSError):
            self.host_pid.unlink(missing_ok=True)


class HostInvocation:
    """Stop a locally spawned CLI by signalling the process group it leads."""

    def __init__(
        self,
        *,
        execution_id: str,
        home: InvocationHome,
        logger: logging.Logger,
        label: str,
        signal_grace: float = SIGNAL_GRACE,
    ):
        self.logger = logger
        self.label = label
        self.signal_grace = signal_grace
        self.host_pid = home.host / f"call-{execution_id}.pid"
        self.pid_file = str(self.host_pid)
        self.group_file = home.host / f"call-{execution_id}{PGID_SUFFIX}"

    def record(self, process: asyncio.subprocess.Process) -> None:
        # Written while the process is alive, because after a backend restart
        # this file is the only thing that can find it again.
        try:
            record_process_group(self.group_file, process.pid)
        except OSError as error:
            self.logger.warning(
                "Could not record the %s process group: %s", self.label, error
            )

    def mark_cancelled(self) -> None:
        """Genuinely nothing to do, and not a placeholder.

        The marker exists because a `docker exec` client can exit before the
        remote shell it asked for has started, leaving a turn running that
        nobody is watching. There is no such window here: asyncio's
        `create_subprocess_exec` returns only once fork and exec have both
        succeeded, so by the time anything can cancel, the process is already
        ours to signal. Writing a marker would create a file nothing reads.
        """

    async def stop(self, process: asyncio.subprocess.Process) -> None:
        for name in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(process.pid), name)
            except ProcessLookupError:
                return
            except OSError as error:  # noqa: PERF203 - later signals must still run
                self.logger.warning(
                    "%s %s signal failed: %s", self.label, name.name, error
                )
            try:
                await asyncio.wait_for(process.wait(), self.signal_grace)
                return
            except TimeoutError:
                pass
        raise RuntimeError(f"{self.label} invocation did not stop after INT/TERM/KILL")

    def remove_pid(self) -> None:
        # The group record goes with it: the turn is over, and leaving it would
        # make a later reap chase a process ID that has already been reused.
        for path in (self.host_pid, self.group_file):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
