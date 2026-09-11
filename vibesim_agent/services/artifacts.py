"""List, classify and resolve artifact files inside one durable workspace.

Browser, interactive Agent, and eval responses expose a stable ``workspace_id``.
Artifact routes use that id to list and download logs, JSON, parquet and plots
from the workspace repo; conversation ids are not filesystem identities.

Two guard levels share one containment check:

  - ``guard_artifact`` — containment only. Backs the token-gated
    ``/api/agent/*/artifacts*`` routes, where an authenticated agent may reach a
    build/VCS tree by exact path on purpose.
  - ``guard_preview`` — containment plus a denylist. Backs the untokened
    browser ``/api/agent/v1/file*`` routes, which must never hand out VCS internals or
    credential-shaped files.

Framework-agnostic on purpose: functions raise plain builtins so the FastAPI
layer can map them to HTTP status codes (mirrors how `eval.run_eval` stays free
of HTTP concerns):
  - FileNotFoundError -> 404 (unknown workspace, missing file, not a file)
  - PermissionError   -> 403 (path escapes the workspace, or is denied)
"""

from __future__ import annotations

import codecs
import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage.registry import WorkspaceRegistry

# Build/cache/VCS trees that are never useful as run artifacts and would blow up
# a listing. Skipped while walking; the token-gated agent routes can still
# download a specific file inside them by exact path if they really need to.
_SKIP_DIRS = {".git", "target", "node_modules", "__pycache__", ".venv", ".uv-cache"}
_DEFAULT_LIMIT = 2000

# Untokened preview refuses these outright. Directory segments reuse _SKIP_DIRS
# (a `.git/config` preview is never a legitimate result artifact); the filename
# patterns cover the credential shapes a repo tree may legitimately contain.
_DENIED_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
    ".npmrc",
    "credentials",
    "credentials.*",
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".ico"}
# Known-binary fast path. Anything not listed here is sniffed instead, so
# extensionless text (Justfile, LICENSE, Dockerfile) still previews as text.
_BINARY_EXTS = {
    ".parquet",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pt",
    ".pth",
    ".safetensors",
    ".npy",
    ".npz",
    ".bin",
    ".so",
    ".dylib",
    ".dll",
    ".a",
    ".o",
    ".rlib",
    ".zip",
    ".gz",
    ".tar",
    ".tgz",
    ".xz",
    ".zst",
    ".whl",
    ".pdf",
    ".nsys-rep",
    ".sqlite-wal",
    ".woff",
    ".woff2",
    ".ttf",
}

# Extension -> highlight.js language id. The UI registers exactly these.
_LANGUAGE_BY_EXT = {
    ".rs": "rust",
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".json": "json",
    ".jsonl": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sh": "bash",
    ".bash": "bash",
    ".sql": "sql",
    ".html": "xml",
    ".xml": "xml",
    ".c": "cpp",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".cu": "cpp",
    ".cuh": "cpp",
}
_LANGUAGE_BY_NAME = {
    "justfile": "makefile",
    "makefile": "makefile",
    "dockerfile": "dockerfile",
}

# A preview is a bounded read, never the whole file. The browser gets a
# truncation flag plus the true size so it can offer a download instead.
MAX_PREVIEW_BYTES = 5 * 1024 * 1024
MAX_PREVIEW_LINES = 100_000
_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class WorkspaceTree:
    """One workspace's filesystem identity.

    ``repo`` is what a relative path resolves against. ``roots`` is what
    containment is checked against: a workspace may declare a logs tree outside
    its repo, and a file living there is still legitimately its own.
    """

    workspace_id: str
    repo: Path
    roots: tuple[Path, ...]

    def relative_to_root(self, resolved: Path) -> Path:
        for root in self.roots:
            if resolved == root:
                return Path(".")
            if root in resolved.parents:
                return resolved.relative_to(root)
        raise PermissionError("path escapes workspace")


def _requested_path(tree: WorkspaceTree, rel_or_abs: str) -> Path:
    requested = Path(rel_or_abs)
    if requested == Path("/workspace") or Path("/workspace") in requested.parents:
        return tree.repo / requested.relative_to("/workspace")
    return requested if requested.is_absolute() else tree.repo / requested


def guard_artifact(tree: WorkspaceTree, rel_or_abs: str) -> Path:
    """Resolve a caller path against the workspace and reject anything outside it.

    Accepts a plain relative path, or a container-absolute `/workspace/...`
    path (agents often copy those straight out of run output).
    """
    requested = _requested_path(tree, rel_or_abs)
    try:
        resolved = requested.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(str(rel_or_abs)) from exc
    tree.relative_to_root(resolved)  # raises PermissionError when outside
    return resolved


def is_denied_name(name: str) -> bool:
    """True when a filename is credential-shaped and must not be previewed."""
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pattern) for pattern in _DENIED_FILE_PATTERNS)


def guard_preview(tree: WorkspaceTree, rel_or_abs: str) -> Path:
    """`guard_artifact` plus the untokened-preview denylist."""
    resolved = guard_artifact(tree, rel_or_abs)
    for path in (_requested_path(tree, rel_or_abs), resolved):
        try:
            relative = tree.relative_to_root(path)
        except PermissionError:
            # An absolute host alias may resolve into an owned root. Its basename
            # still cannot disguise a denied name; target containment was checked.
            relative = Path(".")
        if any(segment in _SKIP_DIRS for segment in relative.parts):
            raise PermissionError("path is inside a build or VCS tree")
        if is_denied_name(path.name):
            raise PermissionError("path looks like a credential file")
    return resolved


def language_for(path: Path) -> str | None:
    """Highlight language id for a path, or None when the UI should not guess."""
    return _LANGUAGE_BY_EXT.get(path.suffix.lower()) or _LANGUAGE_BY_NAME.get(
        path.name.lower()
    )


def preview_kind(path: Path) -> str:
    """Classify a file as ``image``, ``text`` or ``binary``.

    Unknown extensions are sniffed rather than assumed, so extensionless text
    (Justfile, LICENSE) previews and a mislabeled blob does not.
    """
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTS:
        return "image"
    if suffix in _BINARY_EXTS:
        return "binary"
    try:
        with path.open("rb") as handle:
            raw = handle.read(_SNIFF_BYTES + 1)
    except OSError:
        return "binary"
    sample = raw[:_SNIFF_BYTES]
    if b"\0" in sample:
        return "binary"
    try:
        codecs.getincrementaldecoder("utf-8")().decode(
            sample, final=len(raw) <= _SNIFF_BYTES
        )
    except UnicodeDecodeError:
        return "binary"
    return "text"


def read_text_preview(path: Path) -> tuple[str, bool]:
    """Read a bounded UTF-8 preview. Returns (text, truncated)."""
    with path.open("rb") as handle:
        raw = handle.read(MAX_PREVIEW_BYTES + 1)
    truncated = len(raw) > MAX_PREVIEW_BYTES
    if truncated:
        raw = raw[:MAX_PREVIEW_BYTES]
    text = raw.decode("utf-8", errors="replace")
    if truncated:
        # Never end on a half-read line: the UI numbers lines and a torn tail
        # would render as a real one.
        text = text[: text.rfind("\n") + 1] if "\n" in text else text
    lines = text.split("\n")
    if len(lines) > MAX_PREVIEW_LINES:
        text = "\n".join(lines[:MAX_PREVIEW_LINES])
        truncated = True
    return text, truncated


class ArtifactService:
    def __init__(self, registry: WorkspaceRegistry):
        self.registry = registry

    def workspace_tree(self, workspace_id: str) -> WorkspaceTree:
        """Resolve and validate the trees one registered workspace owns."""
        registry = self.registry
        try:
            repo = registry.repo_path(workspace_id).resolve()
        except (KeyError, ValueError) as exc:
            raise FileNotFoundError(f"unknown workspace {workspace_id!r}") from exc
        if not repo.is_dir():
            raise FileNotFoundError(f"workspace repo is missing for {workspace_id!r}")
        roots = [repo]
        try:
            logs = registry.logs_path(workspace_id).resolve()
        except (KeyError, ValueError, OSError):
            logs = None
        # Every workspace today keeps logs inside its repo; listing the tree anyway
        # costs one comparison and avoids a "file exists but 403" if that changes.
        if (
            logs is not None
            and logs.is_dir()
            and logs != repo
            and repo not in logs.parents
        ):
            roots.append(logs)
        return WorkspaceTree(workspace_id=workspace_id, repo=repo, roots=tuple(roots))

    def artifact_meta(self, workspace_id: str, rel_path: str) -> dict[str, Any]:
        """Describe one file or directory with bounded type sniffing."""
        tree = self.workspace_tree(workspace_id)
        resolved = guard_preview(tree, rel_path)
        stat = resolved.stat()
        is_dir = resolved.is_dir()
        if not is_dir and not resolved.is_file():
            raise FileNotFoundError(str(rel_path))
        return {
            "workspace_id": workspace_id,
            "path": str(tree.relative_to_root(resolved)),
            "name": resolved.name,
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "is_dir": is_dir,
            "preview_kind": "directory" if is_dir else preview_kind(resolved),
            "language": None if is_dir else language_for(resolved),
            "preview_byte_limit": MAX_PREVIEW_BYTES,
        }

    def list_artifacts(
        self,
        workspace_id: str,
        *,
        subdir: str | None = None,
        limit: int = _DEFAULT_LIMIT,
        recursive: bool = True,
        preview: bool = False,
    ) -> dict[str, Any]:
        """List a workspace tree (optionally one subdir), path-guarded.

        ``recursive`` walks the whole subtree and reports files only — the shape the
        agent artifact route has always returned. Non-recursive reports one level
        including directories, which is what a browsable preview needs.

        ``preview`` selects the untokened guard and hides denied entries, so a
        browser never lists a file it would then be refused.
        """
        tree = self.workspace_tree(workspace_id)
        base = tree.repo
        guard = guard_preview if preview else guard_artifact
        start = guard(tree, subdir) if subdir else base
        if not start.is_dir():
            raise FileNotFoundError(f"not a directory: {subdir!r}")

        # Report paths relative to whichever root holds `start`, resolved once
        # rather than per walked file.
        walk_root = next(
            root for root in tree.roots if start == root or root in start.parents
        )
        entries: list[dict[str, Any]] = []
        truncated = False
        if recursive:
            # os.walk with in-place dir pruning so heavy trees (target/, .git) are
            # never descended into — rglob would materialize them all first.
            for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
                allowed_dirs = []
                for name in dirnames:
                    path = Path(dirpath) / name
                    try:
                        if (
                            name not in _SKIP_DIRS
                            and guard(tree, str(path)).is_dir()
                            and not path.is_symlink()
                        ):
                            allowed_dirs.append(name)
                    except (OSError, RuntimeError):
                        continue
                dirnames[:] = allowed_dirs
                for name in filenames:
                    path = Path(dirpath) / name
                    try:
                        resolved = guard(tree, str(path))
                        if not resolved.is_file():
                            continue
                        stat = resolved.stat()
                    except (OSError, RuntimeError):
                        continue
                    if len(entries) >= limit:
                        truncated = True
                        break
                    entries.append(
                        {
                            "path": str(path.relative_to(walk_root)),
                            "size": stat.st_size,
                            "mtime": int(stat.st_mtime),
                        }
                    )
                if truncated:
                    break
        else:
            for child in sorted(start.iterdir(), key=lambda item: item.name):
                if child.name in _SKIP_DIRS:
                    continue
                if preview and is_denied_name(child.name):
                    continue
                try:
                    resolved = guard(tree, str(child))
                    is_dir = resolved.is_dir()
                    if not is_dir and not resolved.is_file():
                        continue
                    stat = resolved.stat()
                    kind = "directory" if is_dir else preview_kind(resolved)
                except (OSError, RuntimeError):
                    continue
                if len(entries) >= limit:
                    truncated = True
                    break
                entries.append(
                    {
                        "path": str(child.relative_to(walk_root)),
                        "name": child.name,
                        "size": stat.st_size,
                        "mtime": int(stat.st_mtime),
                        "is_dir": is_dir,
                        "preview_kind": kind,
                    }
                )
        entries.sort(key=lambda entry: (not entry.get("is_dir", False), entry["path"]))
        return {
            "workspace_id": workspace_id,
            "root": str(base),
            "path": str(tree.relative_to_root(start)),
            "count": len(entries),
            "truncated": truncated,
            "files": entries,
        }

    def resolve_artifact(self, workspace_id: str, rel_path: str) -> Path:
        """Resolve one artifact file for download, guarded to stay inside the workspace."""
        resolved = guard_artifact(self.workspace_tree(workspace_id), rel_path)
        if not resolved.is_file():
            raise FileNotFoundError(str(rel_path))
        return resolved

    def resolve_preview(self, workspace_id: str, rel_path: str) -> tuple[Path, str]:
        """Resolve one file for untokened browser preview. Returns (path, kind)."""
        resolved = guard_preview(self.workspace_tree(workspace_id), rel_path)
        if not resolved.is_file():
            raise FileNotFoundError(str(rel_path))
        return resolved, preview_kind(resolved)
