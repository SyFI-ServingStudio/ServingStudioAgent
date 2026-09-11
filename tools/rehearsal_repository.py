"""Prepare an independent Git repository bundle for migration rehearsals.

This is a derived execution copy, not a backup of every Git configuration or
operation state. Files and index entries are preserved; Git remotes, hooks and
host configuration are not activated. Callers must stop working-tree and shared
Git-metadata writers before preparation, then validate non-Git dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.migrate_v1_workspaces import (
    _dependencies,
    _move_entry,
    _overlap,
    _publish,
    _writable_directory,
    _write_json,
)
from tools.migration_files import copy_tree, inventory_tree


class RepositoryPreparationError(ValueError):
    """A repository cannot be prepared with the supported isolation guarantees."""


_CORE_DEFAULTS = {
    "core.filemode": "true",
    "core.ignorecase": "false",
    "core.autocrlf": "false",
    "core.eol": "native",
    "core.symlinks": "true",
    "core.precomposeunicode": "false",
    "core.trustctime": "true",
    "core.checkstat": "default",
}


def _environment():
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_OPTIONAL_LOCKS="0",
        GIT_TERMINAL_PROMPT="0",
        GIT_NO_LAZY_FETCH="1",
    )
    return environment


def _git(repo: Path, *arguments: str, allow_missing=False) -> bytes:
    result = subprocess.run(
        [
            "git",
            "--no-pager",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=" + os.devnull,
            "-c",
            "protocol.allow=never",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(repo),
            *arguments,
        ],
        env=_environment(),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 and not (allow_missing and result.returncode == 1):
        raise RepositoryPreparationError("Git preflight or copy verification failed")
    return result.stdout


def _value(repo, *arguments, allow_missing=False):
    return _git(repo, *arguments, allow_missing=allow_missing).decode().strip()


def _refs(repo):
    rows = _git(
        repo, "for-each-ref", "--format=%(refname)%09%(objectname)%09%(symref)"
    ).decode()
    return [line.split("\t") for line in rows.splitlines()]


def _capture(repo: Path) -> dict:
    if Path(_value(repo, "rev-parse", "--show-toplevel")).resolve() != repo.resolve():
        raise RepositoryPreparationError(
            "nested Git marker does not identify its own repository"
        )
    gitdir = Path(_value(repo, "rev-parse", "--path-format=absolute", "--git-dir"))
    common = Path(
        _value(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    )
    index = Path(
        _value(repo, "rev-parse", "--path-format=absolute", "--git-path", "index")
    )
    for path in (gitdir, common):
        if path.is_symlink() or not path.is_dir():
            raise RepositoryPreparationError("Git metadata must be a real directory")
    for name in (
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "rebase-apply",
        "rebase-merge",
        "BISECT_LOG",
    ):
        if (gitdir / name).exists():
            raise RepositoryPreparationError(
                "finish active Git operations before repository preparation"
            )
    if (
        _git(
            repo,
            "config",
            "--get-regexp",
            r"^extensions\.partialclone$",
            allow_missing=True,
        )
        or _value(
            repo,
            "config",
            "--type=bool",
            "--default",
            "false",
            "--get",
            "core.sparseCheckout",
        )
        == "true"
    ):
        raise RepositoryPreparationError(
            "partial and sparse repositories need explicit preparation"
        )
    if _git(
        repo,
        "config",
        "--get-regexp",
        r"^(filter\.|remote\..*\.promisor$)",
        allow_missing=True,
    ):
        raise RepositoryPreparationError(
            "configured filters or promisor remotes need explicit preparation"
        )
    for key in ("core.excludesFile", "core.attributesFile"):
        if _value(repo, "config", "--get", key, allow_missing=True):
            raise RepositoryPreparationError(
                "external Git rule files need explicit preparation"
            )
    if (common / "objects/info/alternates").exists() or any(
        (common / "objects/pack").glob("*.promisor")
    ):
        raise RepositoryPreparationError(
            "external or promisor object stores are not supported"
        )
    if index.is_symlink() or not index.is_file():
        raise RepositoryPreparationError("repository requires a regular index")
    shared = _value(repo, "rev-parse", "--path-format=absolute", "--shared-index-path")
    if shared and (Path(shared).is_symlink() or not Path(shared).is_file()):
        raise RepositoryPreparationError("split index requires a regular shared index")
    entries = _git(repo, "ls-files", "--stage", "-z")
    if any(
        entry.split(b"\t", 1)[0].split()[-1] != b"0"
        for entry in entries.split(b"\0")
        if entry
    ):
        raise RepositoryPreparationError("unmerged index requires explicit resolution")
    core = {}
    for key, default in _CORE_DEFAULTS.items():
        value = _value(repo, "config", "--default", default, "--get", key)
        if key not in {"core.autocrlf", "core.eol", "core.checkstat"} or (
            key == "core.autocrlf" and value != "input"
        ):
            value = _value(
                repo, "config", "--type=bool", "--default", default, "--get", key
            )
        core[key] = value
    rules = {}
    for name in ("exclude", "attributes"):
        path = Path(
            _value(
                repo,
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "info/" + name,
            )
        )
        if path.is_symlink():
            raise RepositoryPreparationError(
                "Git rule files must not be symbolic links"
            )
        if path.exists():
            if not path.is_file():
                raise RepositoryPreparationError("Git rule files must be regular files")
            rules[name] = {
                "path": path,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return {
        "head": _value(repo, "rev-parse", "--verify", "HEAD"),
        "refs": _refs(repo),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "entries_sha256": hashlib.sha256(entries).hexdigest(),
        "entries": entries,
        "gitdir": gitdir,
        "common": common,
        "index": index,
        "core": core,
        "rules": rules,
        "shared_index": Path(shared) if shared else None,
        "shared_sha256": hashlib.sha256(Path(shared).read_bytes()).hexdigest()
        if shared
        else None,
    }


def _repositories(source: Path, inventory: dict):
    paths = []
    for entry in inventory["entries"]:
        relative = Path(entry["path"])
        if relative.name == ".git" and ".git" not in relative.parts[:-1]:
            if entry["kind"] not in {"directory", "file"}:
                raise RepositoryPreparationError("Git markers must not be symlinks")
            paths.append(relative.parent)
    if Path(".") not in paths:
        raise RepositoryPreparationError("source must have its own Git metadata")
    return {
        path: _capture(source / path)
        for path in sorted(paths, key=lambda path: (len(path.parts), str(path)))
    }


def _working_files(inventory, repositories):
    prefixes = [path / ".git" for path in repositories]
    return [
        {key: value for key, value in entry.items() if key != "hardlink_to"}
        for entry in inventory["entries"]
        if not any(Path(entry["path"]).is_relative_to(prefix) for prefix in prefixes)
    ]


def _private_copy(source: Path, target: Path):
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as original, target.open("xb") as copied:
        shutil.copyfileobj(original, copied, length=1024 * 1024)


def _detach_metadata(source, copied, evidence, snapshot):
    old = copied / ".git"
    old_mtime = copied.stat().st_mtime_ns
    _private_copy(snapshot["index"], evidence / "original-index")
    if snapshot["shared_index"] is not None:
        _private_copy(snapshot["shared_index"], evidence / "original-shared-index")
    with _writable_directory(copied):
        _move_entry(old, evidence / "original-git")
        _git(
            copied,
            "clone",
            "--mirror",
            "--local",
            "--no-hardlinks",
            "--template=",
            str(source),
            str(old),
        )
        # Removing the remote through `git remote remove` can delete mirrored refs.
        _git(
            copied,
            "--git-dir",
            str(old),
            "config",
            "--local",
            "--remove-section",
            "remote.origin",
        )
        _git(copied, "--git-dir", str(old), "config", "--local", "core.bare", "false")
        _git(copied, "config", "--local", "core.hooksPath", os.devnull)
        _git(copied, "config", "--local", "core.fsmonitor", "false")
        for key, value in snapshot["core"].items():
            _git(copied, "config", "--local", key, value)
        for name, rule in snapshot["rules"].items():
            (old / "info").mkdir(exist_ok=True)
            _private_copy(rule["path"], old / "info" / name)
            _private_copy(rule["path"], evidence / ("original-info-" + name))
        for name, _, symbolic in snapshot["refs"]:
            if symbolic:
                _git(copied, "symbolic-ref", name, symbolic)
        _private_copy(snapshot["index"], old / "index")
        if snapshot["shared_index"] is not None:
            _private_copy(snapshot["shared_index"], old / snapshot["shared_index"].name)
            _git(copied, "update-index", "--no-split-index")
        _git(copied, "update-index", "--no-fsmonitor", "--no-untracked-cache")
        _git(copied, "fsck", "--connectivity-only", "--no-reflogs")
    os.utime(copied, ns=(old_mtime, old_mtime))


def _verify_git(repo: Path, snapshot):
    current = _capture(repo)
    if (
        current["head"] != snapshot["head"]
        or current["refs"] != snapshot["refs"]
        or current["entries_sha256"] != snapshot["entries_sha256"]
        or current["core"] != snapshot["core"]
        or {key: value["sha256"] for key, value in current["rules"].items()}
        != {key: value["sha256"] for key, value in snapshot["rules"].items()}
    ):
        raise RepositoryPreparationError(
            "copied HEAD, refs or index entries differ from the source"
        )
    for key in ("gitdir", "common", "index"):
        if not current[key].resolve().is_relative_to(repo.resolve()):
            raise RepositoryPreparationError(
                "copied Git metadata still points outside its repository"
            )
    if _git(repo, "remote"):
        raise RepositoryPreparationError("rehearsal repository must not retain remotes")
    return current


def prepare_repository(source: Path, target: Path, *, source_quiesced=False) -> dict:
    """Create target/repo and target/evidence without changing the original repo."""
    if source_quiesced is not True:
        raise RepositoryPreparationError(
            "confirm that working-tree and shared Git writers have stopped"
        )
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_dir():
        raise RepositoryPreparationError("source must be a real working-tree directory")
    source = source.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError("repository bundle target already exists")
    target = target.parent.resolve(strict=True) / target.name
    if _overlap(source, target):
        raise RepositoryPreparationError(
            "repository source and target must not overlap"
        )
    before = inventory_tree(source)
    repositories = _repositories(source, before)
    for snapshot in repositories.values():
        if _overlap(target, snapshot["gitdir"]) or _overlap(target, snapshot["common"]):
            raise RepositoryPreparationError("target overlaps source Git metadata")
    with TemporaryDirectory(
        prefix=".repository-rehearsal-", dir=target.parent
    ) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir(mode=0o700)
        repo = bundle / "repo"
        if copy_tree(source, repo) != before:
            raise RepositoryPreparationError(
                "source changed after repository preflight"
            )
        evidence = bundle / "evidence"
        evidence.mkdir(mode=0o700)
        git_reports = []
        uninitialized = []
        for relative, snapshot in repositories.items():
            archive = (
                evidence
                / "git"
                / (
                    "root"
                    if relative == Path(".")
                    else hashlib.sha256(str(relative).encode()).hexdigest()
                )
            )
            archive.mkdir(parents=True, mode=0o700)
            _detach_metadata(source / relative, repo / relative, archive, snapshot)
            _verify_git(repo / relative, snapshot)
            for entry in snapshot["entries"].split(b"\0"):
                if entry.startswith(b"160000 "):
                    gitlink = relative / os.fsdecode(entry.split(b"\t", 1)[1])
                    if gitlink not in repositories:
                        uninitialized.append(str(gitlink))
            git_reports.append(
                {
                    "path": str(relative),
                    "head": snapshot["head"],
                    "refs": snapshot["refs"],
                    "index_entries_sha256": snapshot["entries_sha256"],
                    "original_index_sha256": snapshot["index_sha256"],
                    "core": snapshot["core"],
                    "rule_sha256": {
                        key: value["sha256"] for key, value in snapshot["rules"].items()
                    },
                }
            )
        after = inventory_tree(repo)
        if _working_files(after, repositories) != _working_files(before, repositories):
            raise RepositoryPreparationError(
                "working files changed while isolating Git metadata"
            )
        for relative, snapshot in repositories.items():
            if _capture(source / relative) != snapshot:
                raise RepositoryPreparationError(
                    "source Git state changed during preparation"
                )
        if inventory_tree(source) != before:
            raise RepositoryPreparationError(
                "source working tree changed during preparation"
            )
        report = {
            "source": str(source),
            "repo": str(target / "repo"),
            "source_quiesced_by_caller": True,
            "git_verified": True,
            "execution_isolation_verified": False,
            "source_files": before,
            "files": after,
            "repositories": git_reports,
            "uninitialized_gitlinks": sorted(uninitialized),
            "dependencies": _dependencies(repo, source, target / "repo", after),
            "limits": "Derived Git config without remotes/hooks; not an operation-state backup. Validate links, environments and container mounts separately.",
        }
        report = json.loads(json.dumps(report, allow_nan=False))
        _write_json(evidence / "manifest.json", report)

        def verify_published(destination):
            for relative, snapshot in repositories.items():
                _verify_git(destination / "repo" / relative, snapshot)

        _publish(bundle, target, verify=verify_published)
    return report


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--source-quiesced", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = prepare_repository(
            args.source, args.target, source_quiesced=args.source_quiesced
        )
    except (ValueError, OSError) as error:
        parser.exit(
            2,
            f"repository preparation failed ({type(error).__name__}); bundle not accepted\n",
        )
    print(
        json.dumps(
            {
                "repo": report["repo"],
                "git_verified": report["git_verified"],
                "execution_isolation_verified": report["execution_isolation_verified"],
                "repositories": len(report["repositories"]),
                "uninitialized_gitlinks": report["uninitialized_gitlinks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
