"""Readiness of the host PATH, which nothing pins the way the image does.

The container mode gets its guarantees from the runner image: the bootstrap
script asserts that every required tool is present and the Dockerfile installs
exactly one version of each CLI. A host turn inherits whatever is on PATH, so
both properties have to be checked here instead -- the first as a hard failure
before anything is spawned, the second as a warning, because a version the
operator installed deliberately is not a reason to refuse the turn.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable

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
