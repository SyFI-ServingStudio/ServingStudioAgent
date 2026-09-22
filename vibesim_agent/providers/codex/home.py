"""Refresh an isolated Codex profile without importing host conversation state."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ...runtime.invocation import InvocationHome, RoleContext
from ...runtime.permissions import reject_retired_settings

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

    def prepare(self, home: InvocationHome, context: RoleContext) -> None:
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
        # The user's whole `config.toml` is copied in above, so a `sandbox_mode`
        # they added for themselves would silently switch Codex back to the
        # retired permission system and take the profile with it.
        config = home.host / "config.toml"
        existing = config.read_text() if config.is_file() else ""
        reject_retired_settings(existing)
        if context.codex_config:
            reject_retired_settings(context.codex_config)
            config.write_text(
                existing + ("\n" if existing and not existing.endswith("\n") else "")
                + "\n" + context.codex_config
            )
        # `PROFILE_ENTRIES` does not include AGENTS.md, so this slot is free.
        # Codex reads `$CODEX_HOME/AGENTS.md` as global instructions, which is
        # how the role contract reaches a host turn -- additively, alongside the
        # worktree's own AGENTS.md, where a container mount would replace it.
        global_prompt = home.host / "AGENTS.md"
        if global_prompt.is_symlink() or global_prompt.is_file():
            global_prompt.unlink()
        if context.global_prompt is not None:
            shutil.copyfile(context.global_prompt, global_prompt)
        for name in ("sessions", "tmp", "shell_snapshots", "log", "cache"):
            path = home.host / name
            if path.is_symlink():
                raise ValueError("Codex runtime directories must not be symlinks")
            path.mkdir(exist_ok=True)
        self._link_skills(home, context)

    def _link_skills(self, home: InvocationHome, context: RoleContext) -> None:
        """Offer the workspace's skills to Codex as skills, one link each.

        Codex reads `$CODEX_HOME/skills`, and until now nothing put anything
        there: the workspace's own library was reachable only by the path named
        in the role contract, so Codex could `cat` a SKILL.md but never had one
        loaded. Claude has had the link since the legacy backend; the Codex arm
        of that function returned before reaching it.

        A link per skill rather than one link for the directory, because
        `$CODEX_HOME/skills` is *also* where Codex installs its own `.system`
        skills on first run. Pointed at the workspace, that write would land in
        the user's repository — in a worktree, an untracked directory inside a
        real checkout.
        """
        if context.skills_source is None:
            return
        skills = home.host / "skills"
        if skills.is_symlink() or (skills.exists() and not skills.is_dir()):
            skills.unlink()
        skills.mkdir(exist_ok=True)
        offered = {
            entry.name: f"{context.skills.rstrip('/')}/{entry.name}"
            for entry in sorted(context.skills_source.iterdir())
            if entry.is_dir() and (entry / "SKILL.md").is_file()
        } if context.skills_source.is_dir() else {}
        for existing in skills.iterdir():
            # Only links this method made are its to remove. `.system` and
            # anything else Codex owns is a real directory and stays.
            if existing.is_symlink() and offered.get(existing.name) != os.readlink(
                existing
            ):
                existing.unlink()
        for name, target in offered.items():
            link = skills / name
            if not link.is_symlink() and not link.exists():
                link.symlink_to(target, target_is_directory=True)

    @property
    def catalog_filename(self) -> str | None:
        return (
            "models_catalog.json"
            if (self.source / "models_catalog.json").is_file()
            else None
        )
