"""Readiness of the host PATH, which nothing pins the way the image does.

The container mode gets its guarantees from the runner image: the bootstrap
script asserts that every required tool is present and the Dockerfile installs
exactly one version of each CLI. A host turn inherits whatever is on PATH, so
both properties have to be checked here instead -- the first as a hard failure
before anything is spawned, the second as a warning, because a version the
operator installed deliberately is not a reason to refuse the turn.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

# What `docker/runner.Dockerfile` installs. Kept here rather than imported from
# `tools/runner_image.py` because `tools/*` imports `vibesim_agent` and not the
# other way round; `tests/test_runner_image_build.py` pins the three copies
# together.
PINNED_VERSIONS = {"codex": "0.155.1", "claude": "2.1.278"}

_SEMVER = re.compile(r"\b(\d+\.\d+\.\d+)\b")
_VERSION_TIMEOUT = 30

# Keyed by identity of the file, not by name: one `npm i -g` is enough to put a
# different CLI behind the same path, and that is the whole reason to look.
_versions: dict[tuple[str, int, int], str | None] = {}


class HostUnavailable(RuntimeError):
    """A tool this turn needs is not on the host's PATH."""


def check_host_binaries(
    binaries: Iterable[str],
    *,
    logger: logging.Logger,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, str]:
    """Resolve each tool, refusing the turn when one is missing."""
    resolved = {}
    for binary in dict.fromkeys(binaries):
        path = which(binary)
        if path is None:
            raise HostUnavailable(f"host is missing required tool: {binary}")
        resolved[binary] = path
        _report_version(binary, path, logger=logger, run=run)
    return resolved


def _report_version(binary, path, *, logger, run) -> None:
    pinned = PINNED_VERSIONS.get(binary)
    if pinned is None:
        return
    version = _version(path, run=run)
    if version is None:
        logger.warning(
            "Could not read the version of host %s at %s; the runner image pins %s",
            binary,
            path,
            pinned,
        )
    elif version != pinned:
        # Silent otherwise, and not cosmetic: 0.144 and 0.155 of Codex disagree
        # about which permission system is even in effect, so the same
        # conversation can move between two different sandbox postures.
        logger.warning(
            "Host %s is %s but the runner image pins %s; "
            "the two execution modes are not running the same CLI",
            binary,
            version,
            pinned,
        )
    else:
        logger.info("Host %s is %s, matching the runner image", binary, version)


def _version(path, *, run) -> str | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (path, stat.st_mtime_ns, stat.st_size)
    if key not in _versions:
        _versions[key] = _read_version(path, run=run)
    return _versions[key]


PGID_SUFFIX = ".pgid"


def record_process_group(path: Path, pid: int) -> None:
    """Note the group and when its leader started, so a later reap is safe.

    A container is its own reaping unit: whatever a turn left running goes away
    with `docker rm`. A host turn leaves nothing behind but process IDs, and
    those are reused -- so the start time is recorded alongside, and a group is
    only signalled while both still agree.
    """
    pgid = os.getpgid(pid)
    # Read the start time of the group leader, which is what the reap compares.
    started = _started(pgid)
    if started is None:
        return
    path.write_text(f"{pgid} {started}\n")


def reap_process_groups(root: Path, *, logger: logging.Logger) -> int:
    """Kill the groups a previous turn left running under `root`."""
    reaped = 0
    for path in sorted(root.rglob("call-*" + PGID_SUFFIX)):
        try:
            pgid, started = path.read_text().split()
            pgid = int(pgid)
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable process group record %s", path)
            continue
        finally:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        # The leader is gone, so nothing here can be identified safely. Any
        # surviving child of that group now belongs to an unknown pgid, and
        # signalling it on the strength of a reused number is worse than
        # leaving it.
        if _started(pgid) != started:
            continue
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError as error:
            logger.warning("Could not stop orphaned process group %s: %s", pgid, error)
            continue
        logger.warning("Stopped an orphaned host turn's process group %s", pgid)
        reaped += 1
    return reaped


def _started(pid: int) -> str | None:
    """Field 22 of `/proc/<pid>/stat`, read past a comm that may contain ')'."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        return stat[stat.rindex(")") + 1 :].split()[19]
    except (ValueError, IndexError):
        return None


def _read_version(path, *, run) -> str | None:
    try:
        result = run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    found = _SEMVER.search(result.stdout or "")
    return found.group(1) if found else None
