"""List and resolve artifact files inside a per-eval isolated workspace.

An agent runs a prompt through `POST /api/eval`, gets back a `conversation_id`,
then uses that id as `cid` to list and download artifacts (logs, JSON, parquet,
plots) the run produced under `workspaces/<cid>/main`.

Framework-agnostic on purpose: functions raise plain builtins so the FastAPI
layer can map them to HTTP status codes (mirrors how `eval.run_eval` stays free
of HTTP concerns):
  - FileNotFoundError -> 404 (unknown workspace, missing file, not a file)
  - PermissionError   -> 403 (path escapes the workspace)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .codex_runtime.config import workspace_main_for

# Build/cache/VCS trees that are never useful as run artifacts and would blow up
# a listing. Skipped while walking; a caller can still download a specific file
# inside them by exact path if it really needs to.
_SKIP_DIRS = {".git", "target", "node_modules", "__pycache__", ".venv", ".uv-cache"}
_DEFAULT_LIMIT = 2000


def _workspace_root(cid: str) -> Path:
    """Resolve and validate the workspace main tree for one eval conversation."""
    if not cid or "/" in cid or "\\" in cid or cid in (".", ".."):
        raise ValueError("invalid cid")
    base = workspace_main_for(cid)
    if not base.is_dir():
        raise FileNotFoundError(f"no workspace for cid {cid!r}")
    return base.resolve()


def _guard_inside(base: Path, rel_or_abs: str) -> Path:
    """Resolve a user path against `base` and reject anything that escapes it.

    Accepts a plain relative path, or a container-absolute `/workspace/...`
    path (agents often copy those straight out of run output).
    """
    requested = Path(rel_or_abs)
    if requested == Path("/workspace") or Path("/workspace") in requested.parents:
        requested = base / requested.relative_to("/workspace")
    elif not requested.is_absolute():
        requested = base / requested
    try:
        resolved = requested.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(str(rel_or_abs)) from exc
    if resolved != base and base not in resolved.parents:
        raise PermissionError("path escapes workspace")
    return resolved


def list_artifacts(
    cid: str,
    *,
    subdir: str | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> dict[str, Any]:
    """List files under the eval workspace (optionally a subdir), path-guarded."""
    base = _workspace_root(cid)
    start = _guard_inside(base, subdir) if subdir else base
    if not start.is_dir():
        raise FileNotFoundError(f"not a directory: {subdir!r}")

    files: list[dict[str, Any]] = []
    truncated = False
    # os.walk with in-place dir pruning so heavy trees (target/, .git) are never
    # descended into — rglob would materialize them all before we could skip.
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if not path.is_file():  # skip broken symlinks
                continue
            if len(files) >= limit:
                truncated = True
                break
            stat = path.stat()
            files.append(
                {
                    "path": str(path.relative_to(base)),
                    "size": stat.st_size,
                    "mtime": int(stat.st_mtime),
                }
            )
        if truncated:
            break
    files.sort(key=lambda f: f["path"])
    return {
        "cid": cid,
        "root": str(base),
        "count": len(files),
        "truncated": truncated,
        "files": files,
    }


def resolve_artifact(cid: str, rel_path: str) -> Path:
    """Resolve one artifact file for download, guarded to stay inside the workspace."""
    base = _workspace_root(cid)
    resolved = _guard_inside(base, rel_path)
    if not resolved.is_file():
        raise FileNotFoundError(str(rel_path))
    return resolved
