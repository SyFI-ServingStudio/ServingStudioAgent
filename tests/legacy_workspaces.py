"""Legacy descriptor fixtures matching backend/store.py at e26ad6d."""

import json
import os
from pathlib import Path

from tests.legacy_state import create_legacy_database


def create_legacy_workspace(
    root: Path, main: Path, *, workspace_id: str = "w_main", name: str = "Main"
) -> Path:
    directory = root / workspace_id
    directory.mkdir(parents=True, exist_ok=False)
    external = workspace_id == "w_main"
    repo = main if external else directory / "repo"
    descriptor = {
        "schema_version": 1,
        "workspace_id": workspace_id,
        "display_name": name,
        "naming_state": "manual",
        "state": "active",
        "storage_kind": "external" if external else "managed",
        "repo_path": os.path.relpath(repo, directory),
        "logs_path": os.path.relpath(repo / "logs", directory),
        "created_at": 1.0,
        "last_accessed_at": 1.0,
        "base_workspace_id": None if external else "w_main",
        "base_revision": None,
    }
    (directory / "workspace.json").write_text(json.dumps(descriptor))
    index = root / "registry.json"
    registry = (
        json.loads(index.read_text())
        if index.exists()
        else {"schema_version": 1, "workspaces": []}
    )
    registry["workspaces"].append(
        {
            "workspace_id": workspace_id,
            "display_name": name,
            "state": "active",
            "logs_root": os.path.relpath(repo / "logs", root),
        }
    )
    index.write_text(json.dumps(registry))
    return create_legacy_database(directory / "workspace.sqlite")
