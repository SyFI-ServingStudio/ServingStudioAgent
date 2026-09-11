"""Verified offline tree copies for workspace migration.

Writers must already be stopped. Rescanning detects changes; it cannot establish
an instantaneous snapshot. Preserves bytes, mode, mtime and internal hardlinks,
not ownership, atime, ctime, sparse allocation, ACLs or extended attributes.
Symlinks retain their literal targets and may still depend on the old workspace.
The caller must keep the destination private until the entire migration passes.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from contextlib import closing, contextmanager
from pathlib import Path


class TreeMigrationError(ValueError):
    """A tree cannot be copied or verified under the supported fidelity rules."""


_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_CHUNK = 1024 * 1024


def _signature(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _unchanged(before, after):
    if _signature(before) != _signature(after):
        raise TreeMigrationError("source changed during tree migration")


def _hash_file(fd):
    digest = hashlib.sha256()
    while chunk := os.read(fd, _CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


def _scan(root_fd):
    entries = []
    signatures = {}
    inodes = {}
    observed_directories = {".": os.fstat(root_fd)}
    with closing(os.fwalk(".", dir_fd=root_fd, follow_symlinks=False)) as walk:
        for relative, directories, files, directory_fd in walk:
            relative = Path(relative).as_posix()
            info = os.fstat(directory_fd)
            _unchanged(observed_directories.pop(relative), info)
            entries.append(
                {
                    "path": relative,
                    "kind": "directory",
                    "mode": stat.S_IMODE(info.st_mode),
                    "mtime_ns": info.st_mtime_ns,
                }
            )
            signatures[relative] = _signature(info)
            directories.sort()
            for name in sorted(directories + files):
                path = (Path(relative) / name).as_posix()
                before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(before.st_mode):
                    observed_directories[path] = before
                    continue
                entry = {
                    "path": path,
                    "mode": stat.S_IMODE(before.st_mode),
                    "mtime_ns": before.st_mtime_ns,
                }
                if stat.S_ISLNK(before.st_mode):
                    entry.update(
                        kind="symlink", target=os.readlink(name, dir_fd=directory_fd)
                    )
                elif stat.S_ISREG(before.st_mode):
                    fd = os.open(name, _FILE, dir_fd=directory_fd)
                    try:
                        _unchanged(before, os.fstat(fd))
                        digest = _hash_file(fd)
                        _unchanged(before, os.fstat(fd))
                    finally:
                        os.close(fd)
                    entry.update(kind="file", size=before.st_size, sha256=digest)
                    inodes[path] = (before.st_dev, before.st_ino)
                else:
                    raise TreeMigrationError(
                        "special filesystem entries require explicit migration"
                    )
                _unchanged(
                    before, os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                )
                entries.append(entry)
                signatures[path] = _signature(before)
            _unchanged(info, os.fstat(directory_fd))
    if observed_directories:
        raise TreeMigrationError("source directories changed during inventory")
    entries.sort(key=lambda entry: entry["path"])
    canonical = {}
    logical_bytes = unique_bytes = 0
    for entry in entries:
        if entry["kind"] == "file":
            inode = inodes[entry["path"]]
            entry["hardlink_to"] = canonical.get(inode)
            logical_bytes += entry["size"]
            if inode not in canonical:
                canonical[inode] = entry["path"]
                unique_bytes += entry["size"]
    return {
        "entries": entries,
        "logical_bytes": logical_bytes,
        "unique_bytes": unique_bytes,
    }, signatures


@contextmanager
def _open_relative(root_fd, relative, flags):
    """Resolve each parent beneath a held root without following directory links."""
    parts = Path(relative).parts
    parent_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child_fd = os.open(part, _DIRECTORY, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        fd = os.open(parts[-1] if parts else ".", flags, dir_fd=parent_fd)
        try:
            yield fd
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def inventory_tree(source: Path) -> dict:
    """Inventory a real directory; report logical and unique-inode byte counts."""
    fd = os.open(source, _DIRECTORY)
    try:
        return _scan(fd)[0]
    finally:
        os.close(fd)


def _copy_file(source_fd, target_fd):
    with (
        os.fdopen(os.dup(source_fd), "rb") as source,
        os.fdopen(os.dup(target_fd), "wb") as target,
    ):
        shutil.copyfileobj(source, target, length=_CHUNK)


def _metadata(fd, entry):
    os.fchmod(fd, entry["mode"])
    os.utime(fd, ns=(entry["mtime_ns"], entry["mtime_ns"]))


def copy_tree(source: Path, target: Path) -> dict:
    """Copy into an absent target, rescan both trees, and return the manifest.

    A failure removes only the directory inode created by this invocation.
    The parent directory must exist and be controlled by the migration caller.
    """
    source = Path(source)
    target = Path(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    source_resolved = source.resolve(strict=True)
    target_resolved = target.parent.resolve(strict=True) / target.name
    if source_resolved.is_relative_to(
        target_resolved
    ) or target_resolved.is_relative_to(source_resolved):
        raise TreeMigrationError("source and destination trees must not overlap")
    source_fd = os.open(source, _DIRECTORY)
    target_fd = None
    identity = None
    try:
        expected, signatures = _scan(source_fd)
        if shutil.disk_usage(target.parent).free < expected["unique_bytes"]:
            raise TreeMigrationError(
                "insufficient destination space for nonsparse copy"
            )
        target.mkdir(mode=0o700)
        identity = target.lstat()
        target_fd = os.open(target, _DIRECTORY)
        for entry in expected["entries"]:
            relative = entry["path"]
            if relative == ".":
                continue
            destination = target / relative
            if entry["kind"] == "directory":
                destination.mkdir(mode=0o700)
            elif entry["kind"] == "symlink":
                destination.symlink_to(entry["target"])
                os.utime(
                    destination,
                    ns=(entry["mtime_ns"], entry["mtime_ns"]),
                    follow_symlinks=False,
                )
            elif entry["hardlink_to"] is not None:
                os.link(
                    target / entry["hardlink_to"], destination, follow_symlinks=False
                )
            else:
                with _open_relative(source_fd, relative, _FILE) as input_fd:
                    if _signature(os.fstat(input_fd)) != signatures[relative]:
                        raise TreeMigrationError("source file changed before copy")
                    with _open_relative(
                        target_fd,
                        relative,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    ) as output_fd:
                        _copy_file(input_fd, output_fd)
                        _metadata(output_fd, entry)
                    if _signature(os.fstat(input_fd)) != signatures[relative]:
                        raise TreeMigrationError("source file changed during copy")
        directories = [
            entry for entry in expected["entries"] if entry["kind"] == "directory"
        ]
        for entry in sorted(
            directories, key=lambda item: len(Path(item["path"]).parts), reverse=True
        ):
            with _open_relative(target_fd, entry["path"], _DIRECTORY) as fd:
                _metadata(fd, entry)
        actual, _ = _scan(target_fd)
        if actual != expected:
            raise TreeMigrationError("destination tree failed verification")
        if _scan(source_fd) != (expected, signatures):
            raise TreeMigrationError("source tree changed during copy")
        _unchanged(os.fstat(source_fd), source.stat(follow_symlinks=False))
        _unchanged(os.fstat(target_fd), target.stat(follow_symlinks=False))
        return expected
    except BaseException:
        if identity is not None:
            try:
                current = target.lstat()
                if (current.st_dev, current.st_ino) == (
                    identity.st_dev,
                    identity.st_ino,
                ):
                    # Restored directory modes may be read-only when verification fails.
                    for root, dirs, _, fd in os.fwalk(target, follow_symlinks=False):
                        os.fchmod(fd, stat.S_IMODE(os.fstat(fd).st_mode) | 0o700)
                    shutil.rmtree(target)
            except FileNotFoundError:
                pass
        raise
    finally:
        if target_fd is not None:
            os.close(target_fd)
        os.close(source_fd)
