"""Prepare tracked working-tree snapshots inside caller-owned staging trees."""

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path


class WorkspaceSnapshot:
    def __init__(
        self,
        source: Path,
        *,
        process_environment: Mapping[str, str],
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        if not source.is_absolute():
            raise ValueError("workspace source must be absolute")
        self.source = source.resolve()
        self.run = run
        # Ambient Git routing must not redirect initialization into the source.
        self.environment = {
            key: value
            for key, value in process_environment.items()
            if not key.startswith("GIT_")
        }
        self.environment.update(
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_NOSYSTEM="1",
        )

    def _git(self, root: Path, *arguments: str) -> str:
        result = self.run(
            ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *arguments],
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if result.returncode:
            raise RuntimeError("workspace Git operation failed")
        return result.stdout

    def tracked_entries(self) -> tuple[list[Path], list[Path]]:
        files, submodules = [], []
        for entry in self._git(self.source, "ls-files", "--stage", "-z").split("\0"):
            if not entry:
                continue
            metadata, separator, name = entry.partition("\t")
            fields = metadata.split()
            path = Path(name)
            if (
                not separator
                or len(fields) != 3
                or fields[2] != "0"
                or not name
                or path.is_absolute()
                or ".." in path.parts
                or ".git" in path.parts
            ):
                raise ValueError("invalid or unmerged workspace index entry")
            (submodules if fields[0] == "160000" else files).append(path)
        return files, submodules

    def revision(self) -> str | None:
        try:
            return self._git(self.source, "rev-parse", "--verify", "HEAD").strip()
        except RuntimeError:
            return None

    def populate(self, destination: Path) -> Path:
        """Create a new repo; the caller owns publication and failed-stage cleanup."""
        if not destination.is_absolute():
            raise ValueError("workspace destination must be absolute")
        if destination.resolve().is_relative_to(self.source):
            raise ValueError("workspace destination must be outside the source")
        files, _ = self.tracked_entries()
        destination.mkdir(parents=True, exist_ok=False)
        for relative in files:
            source = self.source / relative
            if not source.exists() and not source.is_symlink():
                continue
            target = destination / relative
            if not target.parent.resolve().is_relative_to(destination.resolve()):
                raise ValueError("workspace copy parent escapes destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                resolved = source.resolve(strict=False)
                if resolved.is_relative_to(self.source) or not resolved.exists():
                    target.symlink_to(os.readlink(source))
                elif resolved.is_dir():
                    shutil.copytree(resolved, target, symlinks=True)
                else:
                    shutil.copy2(resolved, target)
            else:
                shutil.copy2(source, target)
        self.ensure_repository(destination)
        return destination

    def ensure_repository(self, destination: Path) -> None:
        """Initialize legacy managed copies only when they have no Git metadata."""
        if not destination.is_absolute() or not destination.is_dir():
            raise ValueError(
                "workspace repository must be an existing absolute directory"
            )
        if destination.resolve().is_relative_to(self.source):
            raise ValueError("cannot initialize the source workspace")
        metadata = destination / ".git"
        if metadata.exists() or metadata.is_symlink():
            return
        self._git(destination, "init", "--template=")
        self._git(destination, "checkout", "-B", "main")
        self._git(destination, "config", "user.name", "ServingStudio UI")
        self._git(destination, "config", "user.email", "vibesim-ui@example.invalid")
        self._git(destination, "add", "-A")
        self._git(
            destination,
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-m",
            "Initial ServingStudioSim workspace snapshot",
        )
