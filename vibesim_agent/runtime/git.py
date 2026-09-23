"""Run Git with ambient routing and user configuration deliberately disabled."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path


class GitError(RuntimeError):
    """A Git invocation failed; stderr is withheld from the caller by design."""


class GitRunner:
    """Every Git call this service makes goes through here.

    The service runs Git against trees it does not own, from a process whose
    environment it does not control. Three ambient inputs can silently redirect
    a command into the wrong repository or run someone else's code, so all three
    are removed rather than trusted:

    - `GIT_DIR`, `GIT_WORK_TREE` and friends retarget a command no matter what
      `-C` says. A worktree operation that picks one up writes to another repo.
    - User and system configuration can set `core.hooksPath`, aliases and
      `safe.directory`. `commit.gpgsign` alone can hang a call on a passphrase
      prompt with no terminal to answer it.
    - Repository hooks run arbitrary code on `commit` and `checkout`.

    Stderr is captured and dropped: it routinely carries absolute paths, remote
    URLs and branch names, and these messages end up in user-facing errors.
    """

    def __init__(
        self,
        process_environment: Mapping[str, str],
        *,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        timeout: float = 120,
    ):
        self.run = run
        self.timeout = timeout
        self.environment = {
            key: value
            for key, value in process_environment.items()
            if not key.startswith("GIT_")
        }
        self.environment.update(
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_NOSYSTEM="1",
            GIT_TERMINAL_PROMPT="0",
        )

    def __call__(self, root: Path, *arguments: str) -> str:
        result = self.run(
            ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *arguments],
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout,
        )
        if result.returncode:
            raise GitError("Git operation failed")
        return result.stdout
