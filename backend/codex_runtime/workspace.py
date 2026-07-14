"""Workspace copy and git bootstrap for each conversation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .commands import run_checked
from .config import LOG, MAIN_DIR, PROMPTS_DIR, agents_prompt_name, workspace_main_for
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
        if target.exists():
            if target.is_dir():
                shutil.copytree(target, dst, symlinks=True)
            else:
                shutil.copy2(target, dst)
        else:
            os.symlink(os.readlink(src), dst)
    else:
        shutil.copy2(src, dst)

def _refresh_workspace_agent_files(workspace_main: Path, *, autonomous: bool) -> None:
    """Copy runtime role instructions into the copied workspace.

    ``main`` currently has no tracked ``AGENTS.md``. Keeping this file inside the
    copied workspace makes shell/tool behavior consistent across orchestrator and
    implementer sessions. ``.codex/skills`` points at the copied repo-local skill
    tree so Codex can discover skills through its native workspace convention.
    """
    prompt_name = agents_prompt_name(autonomous)
    log_event(
        LOG,
        "workspace.refresh_agent_files",
        workspace=str(workspace_main),
        autonomous=autonomous,
        prompt_name=prompt_name,
    )
    shutil.copy2(PROMPTS_DIR / prompt_name, workspace_main / "AGENTS.md")
    codex_dir = workspace_main / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROMPTS_DIR / "orchestrator.schema.json", codex_dir / "orchestrator.schema.json")
    skills_link = codex_dir / "skills"
    if skills_link.exists() or skills_link.is_symlink():
        if skills_link.is_dir() and not skills_link.is_symlink():
            shutil.rmtree(skills_link)
        else:
            skills_link.unlink()
    os.symlink("../skills", skills_link)

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

def prepare_workspace(conversation_id: str, *, autonomous: bool = False) -> Path:
    """Create the per-conversation copy of git-tracked ``main`` files."""
    workspace_main = workspace_main_for(conversation_id)
    if workspace_main.exists():
        log_event(
            LOG,
            "workspace.prepare.reuse",
            conversation_id=conversation_id,
            workspace=str(workspace_main),
        )
        _refresh_workspace_agent_files(workspace_main, autonomous=autonomous)
        _ensure_workspace_git(workspace_main)
        return workspace_main

    log_event(
        LOG,
        "workspace.prepare.create",
        conversation_id=conversation_id,
        workspace=str(workspace_main),
    )
    tmp_root = workspace_main.parent.with_name(workspace_main.parent.name + ".tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_main = tmp_root / "main"
    tmp_main.mkdir(parents=True, exist_ok=True)

    files, gitlinks = _tracked_entries()
    if gitlinks:
        log_event(
            LOG,
            "workspace.skip_submodules",
            conversation_id=conversation_id,
            count=len(gitlinks),
            paths=[str(p) for p in gitlinks],
        )
    for rel_path in files:
        _copy_tracked_file(rel_path, tmp_main)
    _refresh_workspace_agent_files(tmp_main, autonomous=autonomous)
    _ensure_workspace_git(tmp_main)

    workspace_main.parent.parent.mkdir(parents=True, exist_ok=True)
    if workspace_main.parent.exists():
        shutil.rmtree(workspace_main.parent)
    tmp_root.replace(workspace_main.parent)
    return workspace_main
