"""Provision real Git worktrees with the artifacts `worktree add` leaves behind."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .git import GitError, GitRunner

# `git worktree add` materializes committed files only. Three kinds of artifact
# the simulator needs are therefore missing, per
# ServingStudioSim/skills/dev-create-worktree/SKILL.md:
#
# - `profiling/profile.db` is tracked, but the primary tree's working copy is
#   richer than the committed snapshot: GPU runs JIT-fill kernel rows into it
#   without committing. A worktree left with the committed copy re-profiles on
#   its next GPU run.
# - The large workload CSVs under `trace/` are deliberately untracked, so they
#   never reach a new worktree at all and every preset referencing them fails.
# - `.venv/` and `target/` are git-ignored and must NOT be copied: an editable
#   install records an absolute source path, so a copied environment imports
#   Python from the tree it came from.
SKIP_WORKTREE_ARTIFACTS = ("profiling/profile.db",)
UNTRACKED_ARTIFACT_ROOTS = ("trace",)


class WorktreeError(RuntimeError):
    """Provisioning failed; the tree, branch and admin directory are all gone."""


@dataclass(frozen=True)
class Worktree:
    path: Path
    branch: str
    base_revision: str


class WorktreeProvisioner:
    """Create a sibling worktree of `main` and stage its working-copy artifacts.

    Rollback never uses `rmtree`. Removing the directory alone leaves the admin
    directory under `.git/worktrees/<name>`, and Git then refuses that name
    forever -- the one failure mode a user cannot work around by retrying.
    """

    def __init__(
        self,
        main: Path,
        *,
        process_environment: Mapping[str, str],
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        skip_worktree_artifacts: Sequence[str] = SKIP_WORKTREE_ARTIFACTS,
        untracked_artifact_roots: Sequence[str] = UNTRACKED_ARTIFACT_ROOTS,
    ):
        if not main.is_absolute() or not main.is_dir():
            raise ValueError("worktree main checkout must be an existing directory")
        self.main = main.resolve()
        self.run = run
        self.git = GitRunner(process_environment, run=run)
        self.skip_worktree_artifacts = tuple(skip_worktree_artifacts)
        self.untracked_artifact_roots = tuple(untracked_artifact_roots)

    def valid_branch(self, branch: str) -> bool:
        """Ask Git, because the branch arrives from a UI display name over HTTP."""
        try:
            self.git(self.main, "check-ref-format", "--branch", branch)
        except GitError:
            return False
        return True

    def branch_exists(self, branch: str) -> bool:
        try:
            self.git(
                self.main, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"
            )
        except GitError:
            return False
        return True

    def base_revision(self, base: str | None = None) -> str:
        try:
            return self.git(
                self.main, "rev-parse", "--verify", f"{base or 'HEAD'}^{{commit}}"
            ).strip()
        except GitError as error:
            raise WorktreeError("worktree base revision does not resolve") from error

    def create(
        self, destination: Path, *, branch: str, base: str | None = None
    ) -> Worktree:
        if not destination.is_absolute():
            raise ValueError("worktree destination must be absolute")
        if destination.exists() or destination.is_symlink():
            raise WorktreeError("worktree destination already exists")
        if destination.resolve().is_relative_to(self.main):
            # The convention is a sibling of the main checkout; nesting one
            # inside it makes the outer tree permanently dirty.
            raise WorktreeError("worktree must not be nested inside the main checkout")
        if not branch or branch.startswith("-") or not self.valid_branch(branch):
            raise WorktreeError("worktree branch name is not a valid Git ref")
        revision = self.base_revision(base)
        try:
            self.git(
                self.main, "worktree", "add", "-b", branch, str(destination), revision
            )
        except GitError as error:
            raise WorktreeError("worktree branch or destination is already taken") from error
        try:
            self._stage_tracked(destination)
            self._stage_untracked(destination)
        except BaseException:
            self._discard(destination, branch)
            raise
        return Worktree(path=destination, branch=branch, base_revision=revision)

    def _stage_tracked(self, destination: Path) -> None:
        for relative in self.skip_worktree_artifacts:
            source = self.main / relative
            if not source.is_file():
                continue
            self._copy(source, destination / relative)
            try:
                self.git(destination, "update-index", "--skip-worktree", relative)
            except GitError as error:
                # Not cosmetic: without the bit a 60+ MB modified tracked binary
                # sits in the tree, and `git commit -am` sweeps up the primary
                # tree's kernel cache.
                raise WorktreeError(
                    "worktree could not protect a staged tracked artifact"
                ) from error

    def _stage_untracked(self, destination: Path) -> None:
        for root in self.untracked_artifact_roots:
            if not (self.main / root).is_dir():
                continue
            # No `--exclude-standard` on purpose. These CSVs are *ignored*, not
            # merely untracked (`.gitignore` names `trace/aime_long.csv` and
            # `trace/tracelab_*`), so the standard exclusions would filter out
            # every file this step exists to carry and the step would silently
            # stage nothing.
            listing = self.git(self.main, "ls-files", "--others", "-z", "--", root)
            for name in listing.split("\0"):
                if not name:
                    continue
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise WorktreeError("worktree artifact path escapes the checkout")
                source = self.main / relative
                if source.is_file() and not source.is_symlink():
                    self._copy(source, destination / relative)

    def _copy(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Reflink where the filesystem supports it: `profile.db` is 60+ MB and
        # every worktree gets its own.
        result = self.run(
            ["cp", "-a", "--reflink=auto", str(source), str(target)],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if result.returncode or not target.exists():
            raise WorktreeError("worktree could not stage a working-copy artifact")

    def discard(self, worktree: Worktree) -> None:
        """Undo a completed creation; the caller owns whatever failed after it."""
        self._discard(worktree.path, worktree.branch)

    def _discard(self, destination: Path, branch: str) -> None:
        for arguments in (
            ("worktree", "remove", "--force", str(destination)),
            ("worktree", "prune"),
            ("branch", "-D", branch),
        ):
            try:
                self.git(self.main, *arguments)
            except GitError:
                # Best effort: a later step failing must not mask the original
                # provisioning error that sent us here.
                continue
