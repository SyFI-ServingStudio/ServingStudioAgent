"""Refresh an isolated Codex profile without importing host conversation state."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from ...runtime.invocation import InvocationHome

PROFILE_ENTRIES = (
    "auth.json",
    "config.toml",
    "installation_id",
    "version.json",
    "models_cache.json",
    "models_catalog.json",
    ".personality_migration",
    "rules",
)


@dataclass(frozen=True)
class CodexProfile:
    source: Path

    def prepare(self, home: InvocationHome) -> None:
        source = self.source.resolve(strict=True)
        destination = home.host.resolve()
        if not source.is_dir():
            raise ValueError("Codex configuration source must be a directory")
        if (
            destination == source
            or destination.is_relative_to(source)
            or source.is_relative_to(destination)
        ):
            raise ValueError("Codex configuration and runtime homes must be separate")
        if home.host.is_symlink():
            raise ValueError("Codex runtime home must not be a symlink")
        home.host.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in PROFILE_ENTRIES:
            entry, target = source / name, home.host / name
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)
            if entry.is_dir():
                shutil.copytree(entry, target)
            elif entry.is_file():
                shutil.copy2(entry, target)
        for name in ("sessions", "tmp", "shell_snapshots", "log", "cache"):
            path = home.host / name
            if path.is_symlink():
                raise ValueError("Codex runtime directories must not be symlinks")
            path.mkdir(exist_ok=True)

    @property
    def catalog_filename(self) -> str | None:
        return (
            "models_catalog.json"
            if (self.source / "models_catalog.json").is_file()
            else None
        )
