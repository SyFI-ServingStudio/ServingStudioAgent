"""Workspace copy and git bootstrap for each conversation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .commands import run_checked
from .config import LOG, MAIN_DIR, PROMPTS_DIR, workspace_main_for
from ..logging_config import log_event

def _git_tracked_files() -> list[Path]:
    result = run_checked(["git", "-C", str(MAIN_DIR), "ls-files", "-z"], timeout=60)
    return [Path(raw) for raw in result.stdout.split("\0") if raw]

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

def _refresh_workspace_agent_files(workspace_main: Path) -> None:
    """Copy runtime role instructions into the copied workspace.

    ``main`` currently has no tracked ``AGENTS.md``. Keeping this file inside the
    copied workspace makes shell/tool behavior consistent across orchestrator and
    implementer sessions. ``.codex/skills`` points at the copied repo-local skill
    tree so Codex can discover skills through its native workspace convention.
    """
    log_event(LOG, "workspace.refresh_agent_files", workspace=str(workspace_main))
    shutil.copy2(PROMPTS_DIR / "AGENTS.md", workspace_main / "AGENTS.md")
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
    run_checked(["git", "-C", str(workspace_main), "config", "user.name", "MLSim UI"], timeout=60)
    run_checked(
        ["git", "-C", str(workspace_main), "config", "user.email", "mlsim-ui@example.invalid"],
        timeout=60,
    )
    run_checked(["git", "-C", str(workspace_main), "add", "-A"], timeout=60)
    run_checked(
        ["git", "-C", str(workspace_main), "commit", "-m", "Initial MLSim workspace snapshot"],
        timeout=120,
    )

def prepare_workspace(conversation_id: str) -> Path:
    """Create the per-conversation copy of git-tracked ``main`` files."""
    workspace_main = workspace_main_for(conversation_id)
    if workspace_main.exists():
        log_event(
            LOG,
            "workspace.prepare.reuse",
            conversation_id=conversation_id,
            workspace=str(workspace_main),
        )
        _refresh_workspace_agent_files(workspace_main)
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

    for rel_path in _git_tracked_files():
        _copy_tracked_file(rel_path, tmp_main)
    _refresh_workspace_agent_files(tmp_main)
    _ensure_workspace_git(tmp_main)

    workspace_main.parent.parent.mkdir(parents=True, exist_ok=True)
    if workspace_main.parent.exists():
        shutil.rmtree(workspace_main.parent)
    tmp_root.replace(workspace_main.parent)
    return workspace_main
