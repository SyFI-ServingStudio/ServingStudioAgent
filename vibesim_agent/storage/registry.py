"""Discover workspace descriptors and explicitly publish caller-prepared state."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..domain.identifiers import validate_conversation_id
from .database import Database
from .ownership import WorkspaceOwnership

WORKSPACE_ID_PATTERN = re.compile(r"w_[a-zA-Z0-9_-]{1,64}")
SCHEMA_VERSION = 1
NAMING_STATES = ("pending", "generated", "manual")


class WorkspaceRegistry:
    """Discover existing descriptors; registry.json is an external-reader index.

    Construction deliberately does not create the root or register w_main.
    External repository/log paths are explicit descriptor values; registry-owned
    state paths must remain inside the supplied root, including through symlinks.
    """

    def __init__(self, root: Path, *, clock: Callable[[], float] = time.time):
        if not root.is_absolute():
            raise ValueError("workspace registry root must be absolute")
        self._configured_root = root
        self.root = root.resolve()
        self.clock = clock
        self._lock = threading.RLock()

    def initialize_main(
        self, main: Path, *, base_revision: str | None = None
    ) -> dict[str, Any]:
        """Create fresh state only; never adopt or migrate an existing directory."""
        if self._configured_root.exists() or self._configured_root.is_symlink():
            raise FileExistsError("workspace state directory already exists")
        if not main.is_absolute() or not main.is_dir():
            raise ValueError("main checkout must be an existing absolute directory")
        main = main.resolve()
        if self.root.is_relative_to(main):
            raise ValueError("workspace state must be outside the main checkout")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(mode=0o700, exist_ok=False)
        identity = self.root.stat()
        ownership = WorkspaceOwnership(self.root)
        try:
            ownership.acquire()
            destination = self.workspace_dir("w_main")
            destination.mkdir(mode=0o700)
            Database.create(self.database_path("w_main"))
            now = self.clock()
            descriptor = self._validate_descriptor(
                {
                    "schema_version": SCHEMA_VERSION,
                    "workspace_id": "w_main",
                    "display_name": "Main",
                    "naming_state": "manual",
                    "state": "active",
                    "storage_kind": "external",
                    "repo_path": str(main),
                    "logs_path": str(main / "logs"),
                    "created_at": now,
                    "last_accessed_at": now,
                    "base_workspace_id": None,
                    "base_revision": base_revision,
                },
                self.descriptor_path("w_main"),
            )
            self._write_json_atomic(self.descriptor_path("w_main"), descriptor)
            self.rebuild_index()
            return descriptor
        except BaseException:
            try:
                current = self.root.lstat()
            except FileNotFoundError:
                pass
            else:
                if (
                    current.st_dev == identity.st_dev
                    and current.st_ino == identity.st_ino
                ):
                    shutil.rmtree(self.root)
            raise
        finally:
            ownership.close()

    @property
    def registry_path(self) -> Path:
        return self._state_path(self.root / "registry.json")

    def workspace_dir(self, workspace_id: str) -> Path:
        self._validate_workspace_id(workspace_id)
        return self._state_path(self.root / workspace_id)

    def descriptor_path(self, workspace_id: str) -> Path:
        return self._state_path(self.workspace_dir(workspace_id) / "workspace.json")

    def database_path(self, workspace_id: str) -> Path:
        return self._state_path(self.workspace_dir(workspace_id) / "workspace.sqlite")

    def jobs_dir(self, workspace_id: str) -> Path:
        return self._state_path(self.workspace_dir(workspace_id) / "jobs")

    def conversation_runtime_path(
        self, workspace_id: str, conversation_id: str
    ) -> Path:
        self._validate_conversation_id(conversation_id)
        return self._state_path(
            self.workspace_dir(workspace_id) / "runtime" / conversation_id
        )

    def legacy_runtime_path(self, workspace_id: str, conversation_id: str) -> Path:
        """Locate old session state for explicit offline migration only."""
        self._validate_conversation_id(conversation_id)
        return self._state_path(
            self.workspace_dir(workspace_id) / "codex" / conversation_id
        )

    def get(self, workspace_id: str) -> dict[str, Any]:
        with self._lock:
            path = self.descriptor_path(workspace_id)
            if not path.is_file():
                raise KeyError(workspace_id)
            return self._read_descriptor(path)

    def list(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            return self._list(include_archived=include_archived)

    def _list(self, *, include_archived: bool) -> list[dict[str, Any]]:
        descriptors = []
        for path in sorted(self.root.glob("w_*/workspace.json")):
            try:
                descriptor = self.get(path.parent.name)
            except (OSError, ValueError, KeyError):
                continue
            if include_archived or descriptor["state"] == "active":
                descriptors.append(descriptor)
        descriptors.sort(
            key=lambda item: (
                item["workspace_id"] != "w_main",
                -float(item["last_accessed_at"]),
                item["workspace_id"],
            )
        )
        return descriptors

    def repo_path(self, workspace_id: str) -> Path:
        return self._descriptor_value_path(workspace_id, "repo_path")

    def logs_path(self, workspace_id: str) -> Path:
        return self._descriptor_value_path(workspace_id, "logs_path")

    def _descriptor_value_path(self, workspace_id: str, name: str) -> Path:
        with self._lock:
            descriptor = self.get(workspace_id)
            configured = Path(descriptor[name])
            if not configured.is_absolute():
                configured = self.descriptor_path(workspace_id).parent / configured
            return configured.resolve()

    def publish(self, staging: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
        """Publish a prepared managed stage, rolling back our reservation on failure.

        This serializes one registry instance, not independent processes. The
        exclusive directory reservation never replaces a preexisting workspace.
        """
        with self._lock:
            if (
                not staging.is_absolute()
                or staging.is_symlink()
                or staging.parent.resolve() != self.root
                or not staging.name.startswith(".workspace-")
                or not staging.is_dir()
            ):
                raise ValueError(
                    "workspace staging must be a private directory in the registry root"
                )
            if not isinstance(descriptor, dict):
                raise TypeError("invalid workspace descriptor")
            workspace_id = descriptor.get("workspace_id")
            self._validate_workspace_id(workspace_id)
            destination = self.workspace_dir(workspace_id)
            descriptor = self._validate_descriptor(
                dict(descriptor), destination / "workspace.json"
            )
            if (
                descriptor["storage_kind"] != "managed"
                or descriptor["repo_path"] != "repo"
                or descriptor["logs_path"] != "repo/logs"
                or workspace_id == "w_main"
            ):
                raise ValueError(
                    "publication requires a new managed workspace with local repo and logs"
                )
            if (staging / "workspace.json").exists() or (
                staging / "workspace.json"
            ).is_symlink():
                raise ValueError("staging must not contain a published descriptor")
            for name in ("repo", "workspace.sqlite"):
                path = staging / name
                if path.is_symlink() or not (
                    path.is_dir() if name == "repo" else path.is_file()
                ):
                    raise ValueError(
                        "staging requires a repository and initialized database"
                    )
            with Database(staging / "workspace.sqlite").connect():
                pass
            destination.mkdir(exist_ok=False)
            identity = destination.stat()
            try:
                for child in staging.iterdir():
                    child.rename(destination / child.name)
                self._write_json_atomic(destination / "workspace.json", descriptor)
                self.rebuild_index()
            except BaseException:
                current = destination.lstat()
                if (
                    not destination.is_symlink()
                    and current.st_dev == identity.st_dev
                    and current.st_ino == identity.st_ino
                ):
                    shutil.rmtree(destination)
                raise
            return descriptor

    def update(
        self,
        workspace_id: str,
        *,
        display_name: str | None = None,
        state: str | None = None,
        touch: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            descriptor = self.get(workspace_id)
            if display_name is not None:
                descriptor["display_name"] = self._clean_name(display_name)
                descriptor["naming_state"] = "manual"
            if state is not None:
                if state not in ("active", "archived"):
                    raise ValueError("invalid workspace state")
                if workspace_id == "w_main" and state != "active":
                    raise ValueError("w_main cannot be archived")
                descriptor["state"] = state
            if touch:
                descriptor["last_accessed_at"] = self.clock()
            self._validate_descriptor(descriptor, self.descriptor_path(workspace_id))
            self._write_json_atomic(self.descriptor_path(workspace_id), descriptor)
            self.rebuild_index()
            return descriptor

    def apply_generated_name(self, workspace_id: str, name: str) -> bool:
        clean_name = self._clean_name(name)
        with self._lock:
            descriptor = self.get(workspace_id)
            if descriptor["naming_state"] != "pending":
                return False
            descriptor.update(
                display_name=clean_name,
                naming_state="generated",
                last_accessed_at=self.clock(),
            )
            self._validate_descriptor(descriptor, self.descriptor_path(workspace_id))
            self._write_json_atomic(self.descriptor_path(workspace_id), descriptor)
            self.rebuild_index()
            return True

    def rebuild_index(self) -> None:
        """Recover the external-reader index from authoritative descriptors."""
        with self._lock:
            workspaces = [
                {
                    "workspace_id": item["workspace_id"],
                    "display_name": item["display_name"],
                    "state": item["state"],
                    "logs_root": os.path.relpath(
                        self.logs_path(item["workspace_id"]), self.root
                    ),
                }
                for item in self.list(include_archived=True)
            ]
            self._write_json_atomic(
                self.registry_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "workspaces": workspaces,
                },
            )

    @staticmethod
    def _clean_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip() or "\x00" in name:
            raise ValueError("workspace display_name must not be empty")
        return name.strip()

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="." + path.name + ".", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            try:
                stream = os.fdopen(descriptor, "w", encoding="utf-8")
            except BaseException:
                os.close(descriptor)
                raise
            with stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _state_path(self, path: Path) -> Path:
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("workspace state path escapes registry root")
        return path

    def _read_descriptor(self, path: Path) -> dict[str, Any]:
        descriptor = json.loads(path.read_text(encoding="utf-8"))
        return self._validate_descriptor(descriptor, path)

    def _validate_descriptor(
        self, descriptor: dict[str, Any], path: Path
    ) -> dict[str, Any]:
        required = {
            "schema_version",
            "workspace_id",
            "display_name",
            "state",
            "storage_kind",
            "repo_path",
            "logs_path",
            "created_at",
            "last_accessed_at",
        }
        if not isinstance(descriptor, dict) or required - descriptor.keys():
            raise ValueError("workspace descriptor is missing required fields")
        if (
            type(descriptor["schema_version"]) is not int
            or descriptor["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError("unsupported workspace schema_version")
        self._validate_workspace_id(descriptor["workspace_id"])
        if descriptor["workspace_id"] != path.parent.name:
            raise ValueError(
                "workspace descriptor identity does not match its directory"
            )
        for name in ("display_name", "repo_path", "logs_path"):
            value = descriptor[name]
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"invalid workspace {name}")
        if descriptor["state"] not in ("active", "archived"):
            raise ValueError("invalid workspace state")
        if descriptor["storage_kind"] not in ("external", "managed"):
            raise ValueError("invalid workspace storage_kind")
        descriptor.setdefault("naming_state", "manual")
        if descriptor["naming_state"] not in NAMING_STATES:
            raise ValueError("invalid workspace naming_state")
        for name in ("created_at", "last_accessed_at"):
            value = descriptor[name]
            try:
                finite = isinstance(value, (int, float)) and math.isfinite(value)
            except OverflowError:
                finite = False
            if isinstance(value, bool) or not finite:
                raise ValueError(f"invalid workspace {name}")
        return descriptor

    @staticmethod
    def _validate_workspace_id(workspace_id: str) -> None:
        if not isinstance(workspace_id, str) or not WORKSPACE_ID_PATTERN.fullmatch(
            workspace_id
        ):
            raise ValueError("invalid workspace id")

    _validate_conversation_id = staticmethod(validate_conversation_id)
