"""One-time lossless migration from conversation-owned legacy workspaces.

Usage:
    uv run python -m backend.migrate_workspaces --dry-run
    uv run python -m backend.migrate_workspaces --execute

Execution inventories each legacy tree, atomically renames its ``main`` and
``codex-home`` directories into the new workspace envelope, validates the
renamed bytes, imports SQLite rows, then archives the legacy JSON and now-empty
workspace envelope. Rename rollback restores every moved directory if any
validation fails; the migration never duplicates the potentially large trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .codex_runtime.config import LEGACY_WORKSPACES_DIR, UI_DIR
from .store import (
    SERVER_MESSAGE_FIELDS,
    Store,
    WorkspaceRegistry,
    default_workspaces_root,
)

LEGACY_CONVERSATIONS_PATH = UI_DIR / "conversations.json"


@dataclass(frozen=True, slots=True)
class TreeSummary:
    files: int
    bytes: int
    digest: str


def summarize_tree(root: Path) -> TreeSummary:
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    if not root.exists():
        return TreeSummary(files=0, bytes=0, digest=digest.hexdigest())
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
        files += 1
    return TreeSummary(files=files, bytes=total_bytes, digest=digest.hexdigest())


def load_legacy_conversations(path: Path = LEGACY_CONVERSATIONS_PATH) -> list[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text("utf-8"))
    conversations = payload.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"{path} does not contain a conversations list")
    return [conversation for conversation in conversations if isinstance(conversation, dict)]


def migration_plan(
    conversations: Iterable[dict],
    legacy_workspaces_dir: Path = LEGACY_WORKSPACES_DIR,
) -> list[dict]:
    plan = []
    for conversation in conversations:
        conversation_id = str(conversation.get("id") or "")
        if not conversation_id:
            continue
        legacy_workspace = legacy_workspaces_dir / conversation_id
        workspace_id = (
            f"w_legacy_{conversation_id.replace('.', '_')}"
            if legacy_workspace.is_dir()
            else "w_main"
        )
        plan.append(
            {
                "conversation_id": conversation_id,
                "workspace_id": workspace_id,
                "legacy_workspace": str(legacy_workspace)
                if legacy_workspace.is_dir()
                else None,
                "message_count": len(conversation.get("messages") or []),
            }
        )
    return plan


def execute_migration(
    *,
    registry: WorkspaceRegistry,
    conversations_path: Path = LEGACY_CONVERSATIONS_PATH,
    legacy_workspaces_dir: Path = LEGACY_WORKSPACES_DIR,
) -> dict:
    conversations = load_legacy_conversations(conversations_path)
    plan = migration_plan(conversations, legacy_workspaces_dir)
    migration_root = registry.root / "migrations"
    migration_root.mkdir(parents=True, exist_ok=True)
    lock_path = migration_root / ".migration.lock"
    try:
        lock_descriptor = lock_path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(f"migration lock already exists: {lock_path}") from error

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_root = migration_root / timestamp
    imported = 0
    moved_directories: list[tuple[Path, Path]] = []
    created_workspace_ids: list[str] = []
    archived_conversations: Path | None = None
    archived_workspaces: Path | None = None
    try:
        lock_descriptor.write(f"{timestamp}\n")
        lock_descriptor.close()
        archive_root.mkdir(parents=True)

        conversation_by_id = {
            str(conversation["id"]): conversation
            for conversation in conversations
            if conversation.get("id")
        }
        summaries: dict[str, dict[str, TreeSummary]] = {}
        for row in plan:
            legacy_workspace_value = row["legacy_workspace"]
            if legacy_workspace_value is None:
                continue
            source = Path(legacy_workspace_value)
            workspace_id = row["workspace_id"]
            try:
                registry.get(workspace_id)
            except KeyError:
                registry.create(
                    f"Legacy {row['conversation_id']}",
                    workspace_id=workspace_id,
                )
                created_workspace_ids.append(workspace_id)
            else:
                raise RuntimeError(f"migration destination already exists: {workspace_id}")

            workspace_dir = registry.workspace_dir(workspace_id)
            component_summaries: dict[str, TreeSummary] = {}
            source_main = source / "main"
            destination_repo = workspace_dir / "repo"
            if not source_main.is_dir() or destination_repo.exists():
                raise RuntimeError(
                    f"legacy workspace has an invalid main directory: {source}"
                )
            component_summaries["repo"] = summarize_tree(source_main)
            source_main.replace(destination_repo)
            moved_directories.append((destination_repo, source_main))

            source_codex_home = source / "codex-home"
            if source_codex_home.is_dir():
                destination_codex = workspace_dir / "codex" / row["conversation_id"]
                destination_codex.parent.mkdir(parents=True, exist_ok=True)
                component_summaries["codex"] = summarize_tree(source_codex_home)
                source_codex_home.replace(destination_codex)
                moved_directories.append((destination_codex, source_codex_home))

            for component, source_summary in component_summaries.items():
                destination = (
                    destination_repo
                    if component == "repo"
                    else workspace_dir / "codex" / row["conversation_id"]
                )
                if summarize_tree(destination) != source_summary:
                    raise RuntimeError(
                        f"moved workspace failed validation: {row['conversation_id']}:{component}"
                    )
            summaries[workspace_id] = component_summaries
            source.rmdir()

        for row in plan:
            workspace_id = row["workspace_id"]
            store = Store(registry)
            store.import_conversation(
                workspace_id,
                conversation_by_id[row["conversation_id"]],
            )
            imported += 1

        # Validate imported conversation/message counts before archiving inputs.
        for row in plan:
            imported_conversation = Store(registry).get(
                row["workspace_id"],
                row["conversation_id"],
            )
            if imported_conversation is None or len(
                imported_conversation["messages"]
            ) != row["message_count"]:
                raise RuntimeError(
                    f"conversation import failed validation: {row['conversation_id']}"
                )

        legacy_archive = archive_root / "legacy-source"
        legacy_archive.mkdir(parents=True, exist_ok=True)
        if conversations_path.exists():
            archived_conversations = legacy_archive / conversations_path.name
            conversations_path.replace(archived_conversations)
        if legacy_workspaces_dir.exists():
            archived_workspaces = legacy_archive / legacy_workspaces_dir.name
            legacy_workspaces_dir.replace(archived_workspaces)
        marker = {
            "schema_version": 1,
            "completed_at": time.time(),
            "imported_conversations": imported,
            "workspaces": len({row["workspace_id"] for row in plan}),
            "tree_summaries": {
                workspace_id: {
                    component: {
                        "files": summary.files,
                        "bytes": summary.bytes,
                        "sha256": summary.digest,
                    }
                    for component, summary in component_summaries.items()
                }
                for workspace_id, component_summaries in summaries.items()
            },
        }
        (archive_root / "migration.complete.json").write_text(
            json.dumps(marker, indent=2) + "\n",
            "utf-8",
        )
        return marker
    except BaseException:
        # All data movement is same-filesystem rename. Reverse it before
        # removing only the workspace envelopes created by this invocation.
        if archived_workspaces is not None and archived_workspaces.exists():
            archived_workspaces.replace(legacy_workspaces_dir)
        if archived_conversations is not None and archived_conversations.exists():
            archived_conversations.replace(conversations_path)
        for destination, source in reversed(moved_directories):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(source)
        for workspace_id in reversed(created_workspace_ids):
            workspace_dir = registry.workspace_dir(workspace_id)
            for generated_file in [
                workspace_dir / "workspace.sqlite-wal",
                workspace_dir / "workspace.sqlite-shm",
                workspace_dir / "workspace.sqlite",
                workspace_dir / "workspace.json",
            ]:
                generated_file.unlink(missing_ok=True)
            for empty_directory in [
                workspace_dir / "codex",
                workspace_dir,
            ]:
                try:
                    empty_directory.rmdir()
                except OSError:
                    pass
        registry.ensure_main()
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def _authored_messages(messages: list[dict]) -> list[dict]:
    """Strip the fields the server issues, leaving what the legacy file held.

    The comparison this feeds is an integrity check: it refuses to repair a
    conversation whose *content* moved since migration. A message id and its
    turn anchor are assigned by the store and were never in the legacy JSON, so
    including them would make every conversation look changed.
    """
    return [
        {key: value for key, value in message.items() if key not in SERVER_MESSAGE_FIELDS}
        for message in messages
    ]


def repair_completed_timestamps(
    *,
    registry: WorkspaceRegistry,
    archive_root: Path,
) -> dict:
    """Repair the v1 migration's timestamp-only history-order drift.

    Every archived message must still match the imported conversation before
    either timestamp is touched. This prevents a rerun from overwriting
    activity added after the migration completed.
    """
    marker_path = archive_root / "migration.timestamp-repair.json"
    if marker_path.is_file():
        return json.loads(marker_path.read_text("utf-8"))
    conversations_path = archive_root / "legacy-source" / "conversations.json"
    conversations = load_legacy_conversations(conversations_path)
    store = Store(registry)
    workspace_ids = {
        descriptor["workspace_id"]
        for descriptor in registry.list(include_archived=True)
    }
    repaired = 0
    unchanged = 0
    for conversation in conversations:
        conversation_id = str(conversation["id"])
        legacy_workspace_id = f"w_legacy_{conversation_id.replace('.', '_')}"
        workspace_id = (
            legacy_workspace_id
            if legacy_workspace_id in workspace_ids
            else "w_main"
        )
        imported = store.get(workspace_id, conversation_id)
        if imported is None:
            raise RuntimeError(
                f"timestamp repair could not find conversation: {conversation_id}"
            )
        if _authored_messages(imported["messages"]) != conversation.get("messages", []):
            raise RuntimeError(
                "timestamp repair refused a conversation changed after migration: "
                f"{conversation_id}"
            )
        source_created_at = float(conversation["created_at"])
        source_updated_at = float(conversation["updated_at"])
        if (
            imported["created_at"] == source_created_at
            and imported["updated_at"] == source_updated_at
        ):
            unchanged += 1
            continue
        store.restore_imported_timestamps(
            workspace_id,
            conversation_id,
            created_at=source_created_at,
            updated_at=source_updated_at,
        )
        repaired += 1
    result = {
        "schema_version": 1,
        "repaired_at": time.time(),
        "source": str(conversations_path),
        "conversations": len(conversations),
        "repaired": repaired,
        "unchanged": unchanged,
    }
    temporary_marker = marker_path.with_suffix(".json.tmp")
    temporary_marker.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    temporary_marker.replace(marker_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--repair-completed",
        type=Path,
        metavar="MIGRATION_DIR",
    )
    args = parser.parse_args()

    if args.repair_completed is not None:
        result = repair_completed_timestamps(
            registry=WorkspaceRegistry(),
            archive_root=args.repair_completed,
        )
        print(json.dumps(result, indent=2))
        return 0

    conversations = load_legacy_conversations()
    plan = migration_plan(conversations)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "workspaces_root": str(default_workspaces_root()),
                    "conversations": len(plan),
                    "legacy_workspaces": sum(
                        row["legacy_workspace"] is not None for row in plan
                    ),
                    "plan": plan,
                },
                indent=2,
            )
        )
        return 0

    result = execute_migration(registry=WorkspaceRegistry())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
