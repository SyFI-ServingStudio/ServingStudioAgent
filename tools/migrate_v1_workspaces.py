"""Assemble an offline v1 workspace migration in an independent destination.

Dry-run is the default. Applying requires the caller to confirm that all source
writers have stopped; this tool does not stop services, jobs or containers.
Provider resume and external dependencies require separate runtime validation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import stat
from collections.abc import Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.migrate_v1_database import MigrationError, ProviderIdentity, migrate_database
from tools.migration_files import copy_tree, inventory_tree
from vibesim_agent.domain.roles import Role
from vibesim_agent.runtime.homes import role_home
from vibesim_agent.storage.registry import WorkspaceRegistry

_EVIDENCE = ".migration-v1"
_DATABASE_FILES = tuple(
    "workspace.sqlite" + suffix for suffix in ("", "-wal", "-shm", "-journal")
)


def _overlap(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _real_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise MigrationError("migration requires a regular state file")


@contextmanager
def _database_scratch(workspace: Path):
    # SQLite may update a SHM file even with mode=ro. Never open the archive itself.
    with TemporaryDirectory(prefix="agent-migration-db-") as temporary:
        scratch = Path(temporary)
        for name in _DATABASE_FILES:
            path = workspace / name
            if path.exists() or path.is_symlink():
                _real_file(path)
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                with (
                    os.fdopen(descriptor, "rb") as source,
                    (scratch / name).open("xb") as target,
                ):
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        yield scratch / "workspace.sqlite"


def _descriptors(source: Path) -> dict[str, dict]:
    _real_file(source / "registry.json")
    index = json.loads((source / "registry.json").read_text())
    if (
        not isinstance(index, dict)
        or type(index.get("schema_version")) is not int
        or index.get("schema_version") != 1
        or not isinstance(index.get("workspaces"), list)
    ):
        raise MigrationError("unsupported source workspace index")
    registered = []
    for item in index["workspaces"]:
        if not isinstance(item, dict):
            raise MigrationError("invalid source workspace index entry")
        identity = item.get("workspace_id")
        WorkspaceRegistry._validate_workspace_id(identity)
        registered.append(identity)
    registry = WorkspaceRegistry(source)
    descriptors = {}
    for directory in sorted(source.iterdir()):
        path = directory / "workspace.json"
        if not directory.name.startswith("w_") and not path.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise MigrationError("workspace state directories must be real directories")
        _real_file(path)
        _real_file(directory / "workspace.sqlite")
        descriptors[directory.name] = registry.get(directory.name)
    if len(set(registered)) != len(registered) or set(registered) != set(descriptors):
        raise MigrationError("workspace index and disk identities disagree")
    if "w_main" not in descriptors:
        raise MigrationError("source is missing w_main")
    return descriptors


def _mapped_descriptors(source, descriptors, mode, external_paths):
    if mode not in {"production", "rehearsal"}:
        raise MigrationError("migration mode must be production or rehearsal")
    external = {
        key for key, value in descriptors.items() if value["storage_kind"] == "external"
    }
    if (mode == "production" and external_paths) or (
        mode == "rehearsal" and set(external_paths) != external
    ):
        raise MigrationError(
            "rehearsal requires explicit paths for every external workspace"
        )
    old_paths = []
    for identity in external:
        for key in ("repo_path", "logs_path"):
            old_paths.append((source / identity / descriptors[identity][key]).resolve())
    result = {}
    for identity, original in descriptors.items():
        descriptor = dict(original)
        if descriptor["storage_kind"] == "managed":
            if (
                descriptor["repo_path"] != "repo"
                or descriptor["logs_path"] != "repo/logs"
            ):
                raise MigrationError(
                    "nonstandard managed paths require explicit conversion"
                )
        else:
            if mode == "rehearsal" and set(external_paths[identity]) != {
                "repo_path",
                "logs_path",
            }:
                raise MigrationError(
                    "external path override must contain repo_path and logs_path"
                )
            for key in ("repo_path", "logs_path"):
                path = (source / identity / descriptor[key]).resolve()
                if mode == "rehearsal":
                    configured = Path(external_paths[identity][key])
                    if not configured.is_absolute() or not configured.is_dir():
                        raise MigrationError(
                            "rehearsal external paths must be existing absolute directories"
                        )
                    path = configured.resolve()
                    if _overlap(path, source) or any(
                        _overlap(path, old) for old in old_paths
                    ):
                        raise MigrationError(
                            "rehearsal paths overlap original writable state"
                        )
                descriptor[key] = str(path)
            if mode == "rehearsal":
                repo_logs = Path(descriptor["repo_path"]) / "logs"
                if (
                    repo_logs.is_symlink()
                    or not repo_logs.is_dir()
                    or repo_logs.resolve() != Path(descriptor["logs_path"])
                ):
                    raise MigrationError(
                        "rehearsal logs must be the real repo/logs directory mounted in the container"
                    )
        result[identity] = descriptor
    if result["w_main"]["storage_kind"] != "external":
        raise MigrationError("w_main must remain an external workspace")
    return result


def _sessions(database: Path):
    with closing(
        sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    ) as connection:
        connection.execute("PRAGMA query_only=ON")
        return connection.execute(
            "SELECT conversation_id, role, family FROM codex_sessions ORDER BY conversation_id, role"
        ).fetchall()


def _home_plan(source, identity, rows, families, runners):
    plans = []
    workspace = source / identity
    for conversation, role, family in rows:
        WorkspaceRegistry._validate_conversation_id(conversation)
        role = Role(role)
        provider = families[family]
        runner = runners[family]
        old = workspace / "codex" / conversation / role.value
        if runner == "claude":
            old = old / "claude"
        if (
            old.is_symlink()
            or (old.exists() and not old.is_dir())
            or not old.resolve().is_relative_to(workspace.resolve())
        ):
            raise MigrationError("stored session has no contained historical role home")
        # Reject links at every state-owned ancestor, including a contained alias.
        for parent in old.relative_to(workspace).parents:
            if (workspace / parent).is_symlink():
                raise MigrationError(
                    "historical role home ancestors must not be symlinks"
                )
        new = role_home(
            workspace / "runtime" / conversation,
            "/runtime",
            role,
            provider.session_scope,
        ).host
        if (workspace / "runtime").exists() or (workspace / "runtime").is_symlink():
            raise MigrationError(
                "legacy runtime directory conflicts with new active homes"
            )
        shared = []
        if runner == "codex":
            marker = old / ".legacy-shared-runtime-imported"
            if marker.exists() or marker.is_symlink():
                _real_file(marker)
            else:
                for name in ("sessions", "shell_snapshots"):
                    path = old.parent / name
                    if path.exists() or path.is_symlink():
                        if path.is_symlink() or not path.is_dir():
                            raise MigrationError(
                                "shared historical state must be a real directory"
                            )
                        _check_overlay(path, old / name)
                        shared.append(str(path.relative_to(source)))
        if not old.is_dir() and not shared:
            raise MigrationError("stored session has no historical role or shared home")
        plans.append(
            {
                "source": str(old.relative_to(source)),
                "source_exists": old.is_dir(),
                "target": str(new.relative_to(source)),
                "provider_id": provider.provider_id,
                "runner": runner,
                "shared_imports": shared,
            }
        )
    return plans


def _entry_content(entry):
    return {
        key: value
        for key, value in entry.items()
        if key not in {"path", "mtime_ns", "mode", "hardlink_to"}
    }


def _check_overlay(source: Path, target: Path):
    if not target.exists() and not target.is_symlink():
        return
    if target.is_symlink() or not target.is_dir():
        raise MigrationError("shared and role session paths conflict")
    existing = {entry["path"]: entry for entry in inventory_tree(target)["entries"]}
    for entry in inventory_tree(source)["entries"]:
        other = existing.get(entry["path"])
        if other is not None and _entry_content(entry) != _entry_content(other):
            raise MigrationError(
                "shared and role session files conflict; explicit policy required"
            )


@contextmanager
def _writable_directory(path: Path):
    mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode | 0o700)
    try:
        yield
    finally:
        path.chmod(mode)


def _move_entry(source: Path, target: Path):
    mode = stat.S_IMODE(source.lstat().st_mode)
    directory = source.is_dir() and not source.is_symlink()
    if directory:
        source.chmod(mode | 0o700)
    moved = False
    try:
        source.rename(target)
        moved = True
    finally:
        if directory:
            (target if moved else source).chmod(mode)


def _merge_directories(source: Path, target: Path):
    # Both paths are derived copies; the original shared and role trees stay intact.
    with _writable_directory(source), _writable_directory(target):
        for child in source.iterdir():
            destination = target / child.name
            if not destination.exists() and not destination.is_symlink():
                _move_entry(child, destination)
            elif child.is_dir() and not child.is_symlink():
                _merge_directories(child, destination)
            # Equal existing entries were checked before merging. Keep the role copy.


def _activate_home(state: Path, plan: dict) -> dict:
    source = state / plan["source"]
    target = state / plan["target"]
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if plan["source_exists"]:
        copy_tree(source, target)
    else:
        target.mkdir(mode=0o700)
    with _writable_directory(target):
        for shared_relative in plan["shared_imports"]:
            shared = state / shared_relative
            destination = target / shared.name
            _check_overlay(shared, destination)
            if not destination.exists():
                copy_tree(shared, destination)
            else:
                with TemporaryDirectory(
                    prefix=".shared-", dir=target.parent
                ) as temporary:
                    copied = Path(temporary) / "files"
                    copy_tree(shared, copied)
                    _merge_directories(copied, destination)
    return {**plan, "files": inventory_tree(target)}


def _dependencies(
    state: Path, original: Path, final: Path, inventory: dict
) -> list[dict]:
    result = []
    for entry in inventory["entries"]:
        path = state / entry["path"]
        raw = None
        kind = "symlink"
        if entry["kind"] == "symlink":
            raw = entry["target"]
        elif entry["kind"] == "file" and path.name == ".git":
            kind = "gitdir"
            try:
                content = path.read_text().strip() if entry["size"] <= 65536 else ""
            except UnicodeError:
                content = ""
            if not content.startswith("gitdir: "):
                result.append(
                    {
                        "path": entry["path"],
                        "kind": kind,
                        "unrecognized": True,
                        "requires_validation": True,
                    }
                )
                continue
            raw = content[len("gitdir: ") :]
        if raw is None:
            continue
        # Report final lexical paths, not resolution through a temporary tree.
        # Every link needs validation before execution, including internal chains.
        lexical = Path(os.path.abspath(final / Path(entry["path"]).parent / raw))
        result.append(
            {
                "path": entry["path"],
                "kind": kind,
                "target": raw,
                "lexical_target": str(lexical),
                "requires_validation": True,
                "lexically_external": not lexical.is_relative_to(final),
                "points_to_source": lexical.is_relative_to(original),
            }
        )
    return result


def _write_json(path: Path, payload: dict):
    WorkspaceRegistry._write_json_atomic(path, payload)


def _archive(state: Path, identities):
    archive = state / _EVIDENCE / "original"
    archive.mkdir(parents=True, mode=0o700)
    (state / "registry.json").rename(archive / "registry.json")
    for identity in identities:
        original = archive / identity
        original.mkdir(mode=0o700)
        for name in ("workspace.json", *_DATABASE_FILES):
            path = state / identity / name
            if path.exists() or path.is_symlink():
                path.rename(original / name)
    return archive


def _publish(state: Path, target: Path, *, verify=None):
    expected = inventory_tree(state)
    root_mtime = next(
        entry["mtime_ns"] for entry in expected["entries"] if entry["path"] == "."
    )
    target.mkdir(mode=0o700, exist_ok=False)
    owned = target.lstat()
    try:
        # Registry is the last visible artifact. Callers must wait for command success.
        for child in sorted(
            state.iterdir(), key=lambda child: child.name == "registry.json"
        ):
            _move_entry(child, target / child.name)
        os.utime(target, ns=(root_mtime, root_mtime))
        if inventory_tree(target) != expected:
            raise MigrationError("published workspace tree failed verification")
        if verify is not None:
            verify(target)
    except BaseException:
        current = target.lstat()
        if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
            for _, _, _, descriptor in os.fwalk(target, follow_symlinks=False):
                os.fchmod(descriptor, os.fstat(descriptor).st_mode | 0o700)
            shutil.rmtree(target)
        raise


def migrate_workspaces(
    source: Path,
    target: Path,
    *,
    models: Mapping[str, ProviderIdentity],
    families: Mapping[str, ProviderIdentity],
    runners: Mapping[str, str],
    mode: str = "production",
    external_paths: Mapping[str, Mapping[str, Path]] | None = None,
    dry_run: bool = True,
    source_quiesced: bool = False,
) -> dict:
    """Prepare a complete state copy; never start it or validate provider resume."""
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_dir():
        raise MigrationError("source must be a real state directory")
    source = source.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError("migration target already exists")
    target = target.parent.resolve(strict=True) / target.name
    if _overlap(source, target):
        raise MigrationError("source and target workspace roots must not overlap")
    if not dry_run and source_quiesced is not True:
        raise MigrationError(
            "applying requires explicit confirmation that source writers stopped"
        )
    if (source / _EVIDENCE).exists() or (source / _EVIDENCE).is_symlink():
        raise MigrationError(
            "source conflicts with reserved migration evidence directory"
        )
    if set(runners) != set(families) or any(
        runner not in {"codex", "claude"} for runner in runners.values()
    ):
        raise MigrationError(
            "each historical family requires an explicit supported runner"
        )
    original_descriptors = _descriptors(source)
    descriptors = _mapped_descriptors(
        source, original_descriptors, mode, external_paths or {}
    )
    for identity, original in original_descriptors.items():
        if original["storage_kind"] == "external":
            for key in ("repo_path", "logs_path"):
                for path in (
                    (source / identity / original[key]).resolve(),
                    Path(descriptors[identity][key]),
                ):
                    if _overlap(target, path):
                        raise MigrationError(
                            "migration target overlaps an external workspace path"
                        )
    source_files = inventory_tree(source)
    databases, homes = {}, []
    for identity in descriptors:
        with _database_scratch(source / identity) as database:
            report = migrate_database(database, None, models=models, families=families)
            report["source"] = str(source / identity / "workspace.sqlite")
            databases[identity] = report
            homes.extend(
                _home_plan(source, identity, _sessions(database), families, runners)
            )
    report = {
        "format": 1,
        "source": str(source),
        "target": str(target),
        "mode": mode,
        "source_quiesced_by_caller": source_quiesced,
        "verified": False,
        "resume_verified": False,
        "external_paths_verified": False,
        "execution_isolation_verified": False,
        "descriptors": descriptors,
        "databases": databases,
        "homes": homes,
        "source_files": source_files,
        "dependencies": _dependencies(source, source, target, source_files),
        "fidelity": "bytes, mode, mtime, raw symlinks and archive internal hardlinks; not owner, ACL, xattr or sparse allocation",
    }
    if dry_run:
        return json.loads(json.dumps(report, allow_nan=False))
    json.dumps(report, allow_nan=False)
    with TemporaryDirectory(
        prefix=".agent-workspaces-migration-", dir=target.parent
    ) as temporary:
        state = Path(temporary) / "state"
        if copy_tree(source, state) != source_files:
            raise MigrationError("source files changed after preflight")
        state.chmod(0o700)
        workspace_modes = {
            identity: stat.S_IMODE((state / identity).stat().st_mode)
            for identity in descriptors
        }
        for identity, permissions in workspace_modes.items():
            (state / identity).chmod(permissions | 0o700)
        archive = _archive(state, descriptors)
        for identity, descriptor in descriptors.items():
            with _database_scratch(archive / identity) as database:
                converted = migrate_database(
                    database,
                    state / identity / "workspace.sqlite",
                    models=models,
                    families=families,
                )
            if (
                converted["tables"] != databases[identity]["tables"]
                or converted["source_schema"] != databases[identity]["source_schema"]
            ):
                raise MigrationError("source database changed after preflight")
            converted["source"] = str(
                target / _EVIDENCE / "original" / identity / "workspace.sqlite"
            )
            converted["target"] = str(target / identity / "workspace.sqlite")
            databases[identity] = converted
            _write_json(state / identity / "workspace.json", descriptor)
        report["homes"] = [_activate_home(state, plan) for plan in homes]
        registry = WorkspaceRegistry(state)
        registry.rebuild_index()
        # External logs_root values depend on the registry root's final location.
        index = json.loads(registry.registry_path.read_text())
        for item in index["workspaces"]:
            descriptor = descriptors[item["workspace_id"]]
            logs = Path(descriptor["logs_path"])
            if logs.is_absolute():
                item["logs_root"] = os.path.relpath(logs, target)
        _write_json(registry.registry_path, index)
        for identity, permissions in workspace_modes.items():
            (state / identity).chmod(permissions)
        if inventory_tree(source) != source_files:
            raise MigrationError("source tree changed during workspace assembly")
        output = inventory_tree(state)
        report.update(
            verified=True,
            files=output,
            manifest_excludes=[f"{_EVIDENCE}/manifest.json"],
            dependencies=_dependencies(state, source, target, output),
        )
        report = json.loads(json.dumps(report, allow_nan=False))
        evidence_mtime = (state / _EVIDENCE).stat().st_mtime_ns
        _write_json(state / _EVIDENCE / "manifest.json", report)
        os.utime(state / _EVIDENCE, ns=(evidence_mtime, evidence_mtime))
        _publish(state, target)
    return report


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--source-quiesced", action="store_true")
    parser.add_argument(
        "--mode", choices=("production", "rehearsal"), default="production"
    )
    args = parser.parse_args(argv)
    try:
        mapping = json.loads(args.mapping.read_text())
        if set(mapping) - {"models", "families", "runners", "external_paths"}:
            raise ValueError("unknown mapping fields")
        identities = {
            kind: {
                key: ProviderIdentity(**value) for key, value in mapping[kind].items()
            }
            for kind in ("models", "families")
        }
        runners = mapping["runners"]
        external = mapping.get("external_paths", {})
        if (
            not isinstance(runners, dict)
            or any(not isinstance(runner, str) for runner in runners.values())
            or not isinstance(external, dict)
            or any(
                not isinstance(paths, dict)
                or any(not isinstance(path, str) for path in paths.values())
                for paths in external.values()
            )
        ):
            raise ValueError("invalid runners or external paths mapping")
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        parser.error("cannot read a valid workspace migration mapping")
    try:
        report = migrate_workspaces(
            args.source,
            args.target,
            **identities,
            runners=runners,
            external_paths=external,
            mode=args.mode,
            dry_run=not args.apply,
            source_quiesced=args.source_quiesced,
        )
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(
            2,
            f"workspace migration failed ({type(error).__name__}); target not accepted\n",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
