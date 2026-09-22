"""Explicit workspace provisioning; read-only registry construction stays inert."""

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import uuid4

from ..domain.workspaces import WorkspaceKind
from ..runtime.workspace import WorkspaceSnapshot
from ..runtime.worktree import WorktreeError, WorktreeProvisioner
from ..storage.database import Database
from ..storage.registry import NAMING_STATES, SCHEMA_VERSION, WorkspaceRegistry


class WorktreeUnavailable(RuntimeError):
    """Worktree workspaces are not configured on this deployment."""


def _slug(display_name: str) -> str:
    """Derive a branch topic from a display name the user never typed as one.

    The name is auto-generated from the first prompt, so it can be any text at
    all. Git's own validation still runs on the result; this only has to produce
    something that usually passes and never starts with a character Git or a
    shell would read as a flag.
    """
    slug = "".join(
        character if character.isascii() and character.isalnum() else "-"
        for character in display_name.lower()
    )
    slug = "-".join(part for part in slug.split("-") if part)[:40].strip("-")
    return slug or "workspace"


class WorkspaceService:
    def __init__(
        self,
        registry: WorkspaceRegistry,
        snapshot: WorkspaceSnapshot,
        *,
        clock: Callable[[], float] = time.time,
        worktrees: WorktreeProvisioner | None = None,
        worktree_root: Path | None = None,
    ):
        self.registry = registry
        self.snapshot = snapshot
        self.clock = clock
        self.worktrees = worktrees
        self.worktree_root = worktree_root
        self._preparation_lock = RLock()

    @property
    def workspace_kinds(self) -> tuple[WorkspaceKind, ...]:
        """Announced to the UI so it can tell "old server" from "feature off"."""
        if self.worktrees is None or self.worktree_root is None:
            return (WorkspaceKind.COPY,)
        return (WorkspaceKind.COPY, WorkspaceKind.WORKTREE)

    def create(
        self,
        display_name: str,
        *,
        workspace_id: str | None = None,
        naming_state: str = "manual",
    ) -> dict:
        name = display_name.strip()
        if not name or "\0" in name:
            raise ValueError("workspace display_name must not be empty or contain NUL")
        if naming_state not in NAMING_STATES:
            raise ValueError("invalid workspace naming_state")
        workspace_id = workspace_id or f"w_{uuid4().hex[:12]}"
        destination = self.registry.workspace_dir(workspace_id)
        if workspace_id == "w_main" or destination.exists() or destination.is_symlink():
            raise ValueError(f"workspace already exists or is reserved: {workspace_id}")
        now = self.clock()
        descriptor = {
            "schema_version": SCHEMA_VERSION,
            "workspace_id": workspace_id,
            "display_name": name,
            "naming_state": naming_state,
            "state": "active",
            "storage_kind": "managed",
            "repo_path": "repo",
            "logs_path": "repo/logs",
            "created_at": now,
            "last_accessed_at": now,
            "base_workspace_id": "w_main",
            "base_revision": self.snapshot.revision(),
            "workspace_kind": WorkspaceKind.COPY.value,
        }
        self.registry.root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".workspace-", dir=self.registry.root
        ) as temporary:
            staging = Path(temporary)
            self.snapshot.populate(staging / "repo")
            Database.create(staging / "workspace.sqlite")
            return self.registry.publish(staging, descriptor)

    def create_worktree(
        self,
        display_name: str,
        *,
        branch: str | None = None,
        base: str | None = None,
        workspace_id: str | None = None,
        naming_state: str = "manual",
    ) -> dict:
        """Provision a real Git worktree and register it as an external workspace.

        Synchronous on purpose. The staging the background was meant to hide is
        ~50 ms; the cost is `git worktree add`, which cannot move off the request
        anyway, and provisioning is atomic only while it stays one operation.
        """
        if self.worktrees is None or self.worktree_root is None:
            raise WorktreeUnavailable("worktree workspaces are not configured")
        name = display_name.strip()
        if not name or "\0" in name:
            raise ValueError("workspace display_name must not be empty or contain NUL")
        if naming_state not in NAMING_STATES:
            raise ValueError("invalid workspace naming_state")
        workspace_id = workspace_id or f"w_{uuid4().hex[:12]}"
        destination_dir = self.registry.workspace_dir(workspace_id)
        if (
            workspace_id == "w_main"
            or destination_dir.exists()
            or destination_dir.is_symlink()
        ):
            raise ValueError(f"workspace already exists or is reserved: {workspace_id}")
        branch, path = self._reserve_worktree_name(name, branch)
        worktree = self.worktrees.create(path, branch=branch, base=base)
        now = self.clock()
        descriptor = {
            "schema_version": SCHEMA_VERSION,
            "workspace_id": workspace_id,
            "display_name": name,
            "naming_state": naming_state,
            "state": "active",
            "storage_kind": "external",
            "repo_path": str(worktree.path),
            "logs_path": str(worktree.path / "logs"),
            "created_at": now,
            "last_accessed_at": now,
            "base_workspace_id": "w_main",
            "base_revision": worktree.base_revision,
            "workspace_kind": WorkspaceKind.WORKTREE.value,
            "worktree_branch": worktree.branch,
            "worktree_owned": True,
        }
        (worktree.path / "logs").mkdir(exist_ok=True)
        self.registry.root.mkdir(parents=True, exist_ok=True)
        try:
            with TemporaryDirectory(
                prefix=".workspace-", dir=self.registry.root
            ) as temporary:
                staging = Path(temporary)
                Database.create(staging / "workspace.sqlite")
                return self.registry.publish(staging, descriptor)
        except BaseException:
            # The tree exists only to back this descriptor, so a failed
            # publication must not leave one behind with nothing pointing at it.
            self.worktrees.discard(worktree)
            raise

    def _reserve_worktree_name(
        self, display_name: str, branch: str | None
    ) -> tuple[str, Path]:
        root = self.worktree_root
        if branch is not None:
            # Shape first, then availability: a malformed name is a correctable
            # 400, and reporting it as a collision would send the user looking
            # for a branch that does not exist.
            if not self.worktrees.valid_branch(branch):
                raise ValueError(f"worktree branch name is not a valid Git ref: {branch}")
            # An explicitly requested branch is never silently renamed; a
            # collision is the user's to resolve.
            candidate = root / f"wt-{branch}"
            if self.worktrees.branch_exists(branch) or candidate.exists():
                raise FileExistsError(f"worktree branch already exists: {branch}")
            return branch, candidate
        base = _slug(display_name)
        for suffix in range(1, 100):
            topic = base if suffix == 1 else f"{base}-{suffix}"
            candidate = root / f"wt-{topic}"
            if not self.worktrees.branch_exists(topic) and not candidate.exists():
                return topic, candidate
        raise FileExistsError("could not find an unused worktree name")

    def prepare(self, workspace_id: str) -> str:
        """Reuse durable repos; external workspaces are never initialized or copied."""
        with self._preparation_lock:
            descriptor = self.registry.get(workspace_id)
            repository = self.registry.repo_path(workspace_id)
            if workspace_id == "w_main" or descriptor["storage_kind"] == "external":
                if not repository.is_dir():
                    raise ValueError("external workspace repository does not exist")
                return str(repository)
            expected = self.registry.workspace_dir(workspace_id) / "repo"
            if expected.is_symlink() or repository != expected:
                raise ValueError(
                    "managed repository must be the workspace repo directory"
                )
            if repository.exists():
                self.snapshot.ensure_repository(repository)
                return str(repository)
            # Older descriptors can precede preparation. Reserve without replacing
            # another creator, then move only our staged entries into that directory.
            with TemporaryDirectory(
                prefix=".workspace-", dir=self.registry.root
            ) as temporary:
                prepared = self.snapshot.populate(Path(temporary) / "repo")
                repository.mkdir(exist_ok=False)
                try:
                    for child in prepared.iterdir():
                        child.rename(repository / child.name)
                except BaseException:
                    shutil.rmtree(repository)
                    raise
            return str(repository)
