"""The Codex permission posture, which is per workspace and not a constant.

Codex 0.155 has two permission generations and they do not compose: whenever
`sandbox_mode` or `--sandbox` appears, `default_permissions` is silently
ignored and the older `[sandbox_workspace_write]` settings take over. A sandbox
that fails open without saying so is the worst outcome available here, so the
new system is used exclusively and the old keys are refused on sight.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

HOST_PROFILE = "vibesim_host"
# Keys from the retired generation. Either one turns the profile off.
RETIRED_KEYS = ("sandbox_mode", "sandbox_workspace_write")


@dataclass(frozen=True)
class CodexPermissions:
    """What the turn passes on the command line, plus the profile it names."""

    profile: str
    approval_policy: str
    reviewer: str | None = None
    # TOML appended to the managed `$CODEX_HOME/config.toml`; empty when the
    # profile is one of Codex's own built-ins.
    config: str = ""


# Docker is the boundary in a container, and Codex's Linux sandbox is a bundled
# bubblewrap that needs unprivileged user namespaces -- not something to rely on
# inside a container. `never` is set explicitly so the model is told not to ask:
# there is nothing left to escalate to, and requesting it would only burn turns.
CONTAINER_PERMISSIONS = CodexPermissions(
    profile=":danger-full-access", approval_policy="never"
)


def host_permissions(
    *,
    git_dir: Path,
    git_common_dir: Path,
    workspace_roots: Iterable[Path] = (),
    unix_sockets: Iterable[str] = (),
) -> CodexPermissions:
    """Build the per-workspace profile that lets a worktree turn commit.

    `:workspace` leaves `.git` read-only, and in a worktree `.git` is a *file*
    pointing outside the tree, so the documented
    `filesystem.":workspace_roots".".git"` override does not reach it -- a real
    worktree still fails with `index.lock: Read-only file system`. Both git
    directories therefore have to be granted by absolute path, and both have to
    be computed from the repository, which is what makes this per workspace
    rather than a static config file.

    The cost is real and cannot be fixed here: the common directory is shared by
    every worktree, so an agent that can commit can also rewrite shared refs and
    objects and delete other branches.
    """
    lines = [
        f"[permissions.{HOST_PROFILE}]",
        'extends = ":workspace"',
        "network.enabled = true",
    ]
    for socket in unix_sockets:
        lines.append(f"network.unix_sockets.{_key(socket)} = \"allow\"")
    for root in _unique(workspace_roots):
        lines.append(f"workspace_roots.{_key(str(root))} = true")
    # In a plain checkout these two are the same directory, and TOML rejects a
    # duplicate key outright -- the whole profile fails to load.
    for directory in _unique((git_dir, git_common_dir)):
        lines.append(f"filesystem.{_key(str(directory))} = \"write\"")
    return CodexPermissions(
        profile=HOST_PROFILE,
        # `on-request` with `auto_review` is the only workable pair in a headless
        # service: `codex exec` otherwise defaults to asking a user who is not
        # there. The profile is deliberately wide so the everyday path never
        # reaches the reviewer -- an approved escalation runs with no sandbox at
        # all, so frequent escalation means the profile is missing something.
        approval_policy="on-request",
        reviewer="auto_review",
        config="\n".join(lines) + "\n",
    )


def validated(permissions: CodexPermissions) -> CodexPermissions:
    """Refuse a posture that would read like a boundary without being one."""
    if not permissions.profile:
        raise ValueError("Codex permission profile is required")
    if permissions.approval_policy not in {"never", "on-request"}:
        # `untrusted` is unsupported and `on-failure` deprecated since 0.155.
        raise ValueError("unsupported Codex approval policy")
    if permissions.reviewer not in {None, "user", "auto_review"}:
        raise ValueError("unsupported Codex approvals reviewer")
    if permissions.reviewer is not None and permissions.approval_policy == "never":
        # `never` tells the model not to request escalation at all, so a
        # reviewer would never be consulted. Rejecting the pair keeps a dead
        # setting from reading like an active boundary.
        raise ValueError("Codex approvals reviewer requires an approval policy")
    if permissions.reviewer is None and permissions.approval_policy == "on-request":
        # `codex exec` would then default to `user`, in a service where there
        # is nobody to ask: every escalation would hang or be refused.
        raise ValueError("Codex on-request approvals require a reviewer")
    if permissions.config and not permissions.profile.startswith(HOST_PROFILE):
        raise ValueError("Codex profile configuration must name its own profile")
    return permissions


def reject_retired_settings(config: str) -> None:
    """Refuse a user config that would silently disable the profile."""
    for line in config.splitlines():
        stripped = line.strip().lstrip("[").replace('"', "").replace("'", "")
        for key in RETIRED_KEYS:
            if stripped.startswith(key):
                raise ValueError(
                    f"Codex configuration sets {key}, which silently disables "
                    "the permission profile; remove it from the profile source"
                )


def _key(value: str) -> str:
    # Codex refuses a relative `filesystem` entry outright, and a newline would
    # produce a profile that does not parse -- either way the boundary is gone,
    # so both are caught here rather than at the far end.
    if not value.startswith("/") or "\n" in value:
        raise ValueError(
            f"Codex permission paths must be absolute single lines: {value!r}"
        )
    # TOML basic strings accept JSON's escaping for everything appearing here.
    return json.dumps(value, ensure_ascii=False)


def _unique(values):
    return list(dict.fromkeys(values))
