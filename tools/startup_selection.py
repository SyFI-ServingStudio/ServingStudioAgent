"""Publish and validate the migrated state selected by a managed startup.

Callers hold migration/startup ownership and keep old writers stopped. These
records select a verified copy; they neither stop writers nor authorize rollback.
"""

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from tools.migrate_v1_database import MigrationError, _mapping_manifest
from tools.migrate_v1_workspaces import _descriptors
from tools.migration_files import inventory_tree
from tools.startup_state import inspect_state

_MANIFEST = ".migration-v1/manifest.json"


def _root(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise MigrationError("selected state requires an absolute real directory")
    path = path.resolve(strict=True)
    info = path.stat()
    return {"path": str(path), "device": info.st_dev, "inode": info.st_ino}


def _location(path, source, target):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise MigrationError("startup selection requires an absolute non-symlink file")
    path = path.parent.resolve(strict=True) / path.name
    if (
        source == target
        or source.is_relative_to(target)
        or target.is_relative_to(source)
    ):
        raise MigrationError("selected source and target must not overlap")
    if path.is_relative_to(source) or path.is_relative_to(target):
        raise MigrationError(
            "startup selection must be outside source and target state"
        )
    return path


def _read(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise MigrationError("startup evidence must be a regular file")
        raw = stream.read()
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise MigrationError("startup evidence contains invalid JSON") from error
    if not isinstance(value, dict):
        raise MigrationError("startup evidence must contain a JSON object")
    return raw, value


def _report(source, target, mapping, mode, providers):
    evidence = target / ".migration-v1"
    if evidence.is_symlink() or not evidence.is_dir():
        raise MigrationError("selected target is missing real migration evidence")
    raw, report = _read(target / _MANIFEST)
    if (
        type(report.get("format")) is not int
        or report["format"] != 1
        or report.get("source") != str(source)
        or report.get("target") != str(target)
        or mode not in {"production", "rehearsal"}
        or report.get("mode") != mode
        or report.get("verified") is not True
        or report.get("source_quiesced_by_caller") is not True
        or report.get("manifest_excludes") != [_MANIFEST]
    ):
        raise MigrationError(
            "selected target lacks a matching successful migration report"
        )
    databases = report.get("databases")
    descriptors = report.get("descriptors")
    if (
        not isinstance(databases, dict)
        or not databases
        or not isinstance(descriptors, dict)
        or set(databases) != set(descriptors)
        or any(
            not isinstance(item, dict)
            or item.get("verified") is not True
            or item.get("mapping") != mapping
            for item in databases.values()
        )
    ):
        raise MigrationError(
            "selected migration database mapping does not match configuration"
        )
    if (
        not isinstance(report.get("files"), dict)
        or not isinstance(report["files"].get("entries"), list)
        or any(
            not isinstance(descriptor, dict)
            or (
                descriptor.get("storage_kind") == "external"
                and any(
                    not isinstance(descriptor.get(key), str)
                    for key in ("repo_path", "logs_path")
                )
            )
            for descriptor in descriptors.values()
        )
    ):
        raise MigrationError(
            "selected migration report contains invalid file or workspace records"
        )
    homes = report.get("homes")
    if not isinstance(homes, list) or any(
        not isinstance(home, dict)
        or not isinstance(home.get("provider_id"), str)
        or home.get("provider_id") not in providers
        or home.get("runner") != providers[home["provider_id"]]
        for home in homes
    ):
        raise MigrationError(
            "selected migration runner mapping does not match configuration"
        )
    return hashlib.sha256(raw).hexdigest(), report


def _mapping(models, families):
    return {
        "models": _mapping_manifest(models),
        "families": _mapping_manifest(families, allow_empty=True),
    }


def _providers(families, runners):
    if set(families) != set(runners) or any(
        runner not in {"codex", "claude"} for runner in runners.values()
    ):
        raise MigrationError("each historical family requires a supported runner")
    providers = {}
    for family, identity in families.items():
        runner = runners[family]
        if providers.setdefault(identity.provider_id, runner) != runner:
            raise MigrationError("provider has conflicting historical runners")
    return providers


def _outside_repositories(path, source, report):
    descriptors = [*report["descriptors"].items(), *_descriptors(source).items()]
    for identity, descriptor in descriptors:
        if descriptor.get("storage_kind") == "external":
            for key in ("repo_path", "logs_path"):
                root = (source / identity / descriptor[key]).resolve()
                if path.is_relative_to(root):
                    raise MigrationError(
                        "startup selection must be outside external workspace paths"
                    )


def publish_selection(
    path, source, target, *, models, families, runners, mode="production"
):
    """Select an unchanged, verified migration output without overwriting a record."""
    source_identity, target_identity = _root(source), _root(target)
    source, target = Path(source_identity["path"]), Path(target_identity["path"])
    path = _location(path, source, target)
    mapping = _mapping(models, families)
    digest, report = _report(
        source, target, mapping, mode, _providers(families, runners)
    )
    _outside_repositories(path, source, report)
    expected = report.get("files", {}).get("entries")
    if inventory_tree(source) != report.get("source_files"):
        raise MigrationError("selected migration source changed before publication")
    actual = [
        entry
        for entry in inventory_tree(target)["entries"]
        if entry["path"] != _MANIFEST
    ]
    if actual != expected or inspect_state(target) != "current":
        raise MigrationError("selected migration target changed before publication")
    if _root(source) != source_identity or _root(target) != target_identity:
        raise MigrationError("selected state directory identity changed")
    record = {
        "format": 1,
        "source": source_identity,
        "target": target_identity,
        "manifest_sha256": digest,
        "mapping": mapping,
        "runners": runners,
        "mode": mode,
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=".agent-startup-selection-", dir=path.parent
    )
    try:
        try:
            stream = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            stream.write((json.dumps(record, sort_keys=True, indent=2) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def load_selection(path, source, *, models, families, runners, mode="production"):
    """Return a selected current root, or None only when no record exists.

    Normal new-version writes may change target contents. Never compare the live
    target with its initial inventory or fall back to the old state on failure.
    """
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise MigrationError("startup selection requires an absolute non-symlink file")
    try:
        _, record = _read(path)
    except FileNotFoundError:
        return None
    if set(record) != {
        "format",
        "source",
        "target",
        "manifest_sha256",
        "mapping",
        "runners",
        "mode",
    }:
        raise MigrationError("unsupported startup selection record")
    if type(record["format"]) is not int or record["format"] != 1:
        raise MigrationError("unsupported startup selection version")
    for identity in (record["source"], record["target"]):
        if (
            not isinstance(identity, dict)
            or set(identity) != {"path", "device", "inode"}
            or not isinstance(identity["path"], str)
            or type(identity["device"]) is not int
            or type(identity["inode"]) is not int
        ):
            raise MigrationError("invalid startup selection directory identity")
    source_identity = _root(source)
    selected = record["target"]
    if not isinstance(selected, dict) or not isinstance(selected.get("path"), str):
        raise MigrationError("invalid selected target identity")
    target_identity = _root(selected["path"])
    source, target = Path(source_identity["path"]), Path(target_identity["path"])
    path = _location(path, source, target)
    mapping = _mapping(models, families)
    if (
        record["source"] != source_identity
        or selected != target_identity
        or record["mapping"] != mapping
        or record["runners"] != runners
        or record["mode"] != mode
    ):
        raise MigrationError("startup selection identity or configuration changed")
    digest, report = _report(
        source, target, mapping, mode, _providers(families, runners)
    )
    _outside_repositories(path, source, report)
    if record["manifest_sha256"] != digest:
        raise MigrationError("selected migration report changed")
    if inspect_state(target) != "current":
        raise MigrationError("selected state is not the current format")
    return target
