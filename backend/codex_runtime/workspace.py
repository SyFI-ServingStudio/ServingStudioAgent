"""Managed workspace copy and git bootstrap.

A workspace is durable shared working state. Conversations no longer create
their own repo copies; they reuse the repo selected by ``workspace_id``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .commands import run_checked
from .config import LOG, MAIN_DIR, workspace_main_for
from ..logging_config import log_event

def _tracked_entries() -> tuple[list[Path], list[Path]]:
    """Return (files, gitlinks) tracked in ``main``.

    ``git ls-files`` lists submodule gitlinks (mode ``160000``) as entries, but
    they are directories on disk and cannot be copied as files, so callers copy
    only ``files`` and skip ``gitlinks``.
    """
    result = run_checked(["git", "-C", str(MAIN_DIR), "ls-files", "-s", "-z"], timeout=60)
    files: list[Path] = []
    gitlinks: list[Path] = []
    for raw in result.stdout.split("\0"):
        if not raw:
            continue
        meta, _, path = raw.partition("\t")
        mode = meta.split(" ", 1)[0]
        (gitlinks if mode == "160000" else files).append(Path(path))
    return files, gitlinks


def main_submodule_paths() -> list[Path]:
    """Relative paths of submodule gitlinks tracked in ``main`` that are checked
    out on disk.

    The workspace copy skips them (they are directories, not files, and the vLLM
    checkout alone is ~5 GB); the docker layer instead bind-mounts them read-only
    at the same path, so they are shared across conversations, not duplicated.
    """
    _files, gitlinks = _tracked_entries()
    return [p for p in gitlinks if (MAIN_DIR / p).is_dir()]

def _copy_tracked_file(rel_path: Path, dst_root: Path) -> None:
    src = MAIN_DIR / rel_path
    dst = dst_root / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if src.is_symlink():
        target = src.resolve(strict=False)
        try:
            target.relative_to(MAIN_DIR.resolve())
            target_is_inside_main = True
        except ValueError:
            target_is_inside_main = False

        # Repo-internal links such as `.codex/skills -> ../skills` are part of
        # the workspace structure and must remain links. Dereferencing them
        # creates a stale second skill tree that no longer follows the
        # canonical `skills/` directory. External links such as `old-doc` still
        # need materializing because their targets are not copied into managed
        # workspaces.
        if target_is_inside_main:
            os.symlink(os.readlink(src), dst)
        elif target.exists():
            if target.is_dir():
                shutil.copytree(target, dst, symlinks=True)
            else:
                shutil.copy2(target, dst)
        else:
            os.symlink(os.readlink(src), dst)
    else:
        shutil.copy2(src, dst)


def _copy_existing_tracked_files(
    files: list[Path],
    destination_root: Path,
    *,
    workspace_id: str,
) -> None:
    """Copy the tracked working-tree snapshot, preserving tracked deletions.

    ``git ls-files`` describes the index, so it still returns paths deleted in
    a dirty working tree. Those paths must remain absent in the managed
    workspace rather than turning a valid tracked deletion into a 500.
    """
    deleted_paths: list[str] = []
    for rel_path in files:
        source = MAIN_DIR / rel_path
        if not source.exists() and not source.is_symlink():
            deleted_paths.append(str(rel_path))
            continue
        _copy_tracked_file(rel_path, destination_root)
    if deleted_paths:
        log_event(
            LOG,
            "workspace.skip_deleted_tracked",
            workspace_id=workspace_id,
            count=len(deleted_paths),
            paths=deleted_paths,
        )


def _ensure_workspace_git(workspace_main: Path) -> None:
    """Make the copied workspace a local git repo for branch/commit hygiene."""
    if (workspace_main / ".git").exists():
        log_event(LOG, "workspace.git.exists", workspace=str(workspace_main))
        return

    log_event(LOG, "workspace.git.init", workspace=str(workspace_main))
    run_checked(["git", "-C", str(workspace_main), "init"], timeout=60)
    run_checked(["git", "-C", str(workspace_main), "checkout", "-B", "main"], timeout=60)
    run_checked(["git", "-C", str(workspace_main), "config", "user.name", "VibeSim UI"], timeout=60)
    run_checked(
        ["git", "-C", str(workspace_main), "config", "user.email", "vibesim-ui@example.invalid"],
        timeout=60,
    )
    run_checked(["git", "-C", str(workspace_main), "add", "-A"], timeout=60)
    run_checked(
        ["git", "-C", str(workspace_main), "commit", "-m", "Initial VibeSim workspace snapshot"],
        timeout=120,
    )

def prepare_workspace(workspace_id: str) -> Path:
    """Create a managed workspace's one tracked-file copy.

    ``w_main`` is the external development checkout and is never rewritten by
    this helper. The tracked, blank ``AGENTS.md`` is only a stable bind target;
    Docker overlays the selected conversation contract read-only, so
    conversation settings cannot dirty the shared repo.
    """
    workspace_main = workspace_main_for(workspace_id)
    if workspace_id == "w_main":
        if not workspace_main.is_dir():
            raise RuntimeError(f"main workspace does not exist: {workspace_main}")
        return workspace_main
    if workspace_main.exists():
        log_event(
            LOG,
            "workspace.prepare.reuse",
            workspace_id=workspace_id,
            workspace=str(workspace_main),
        )
        _ensure_workspace_git(workspace_main)
        return workspace_main

    log_event(
        LOG,
        "workspace.prepare.create",
        workspace_id=workspace_id,
        workspace=str(workspace_main),
    )
    tmp_root = workspace_main.with_name(workspace_main.name + ".tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_main = tmp_root
    tmp_main.mkdir(parents=True, exist_ok=True)

    files, gitlinks = _tracked_entries()
    if gitlinks:
        log_event(
            LOG,
            "workspace.skip_submodules",
            workspace_id=workspace_id,
            count=len(gitlinks),
            paths=[str(p) for p in gitlinks],
        )
    _copy_existing_tracked_files(files, tmp_main, workspace_id=workspace_id)
    _ensure_workspace_git(tmp_main)

    workspace_main.parent.mkdir(parents=True, exist_ok=True)
    if workspace_main.exists():
        shutil.rmtree(workspace_main)
    tmp_root.replace(workspace_main)
    return workspace_main
