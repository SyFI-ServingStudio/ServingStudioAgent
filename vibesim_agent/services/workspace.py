"""Explicit workspace provisioning; read-only registry construction stays inert."""

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import uuid4

from ..runtime.workspace import WorkspaceSnapshot
from ..storage.database import Database
from ..storage.registry import NAMING_STATES, SCHEMA_VERSION, WorkspaceRegistry


class WorkspaceService:
    def __init__(
        self,
        registry: WorkspaceRegistry,
        snapshot: WorkspaceSnapshot,
        *,
        clock: Callable[[], float] = time.time,
    ):
        self.registry = registry
        self.snapshot = snapshot
        self.clock = clock
        self._preparation_lock = RLock()

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
        }
        self.registry.root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".workspace-", dir=self.registry.root
        ) as temporary:
            staging = Path(temporary)
            self.snapshot.populate(staging / "repo")
            Database.create(staging / "workspace.sqlite")
            return self.registry.publish(staging, descriptor)

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
