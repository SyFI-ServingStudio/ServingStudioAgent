"""Offer skills to a CLI one link at a time, in a directory it also writes."""

import os
from collections.abc import Mapping
from pathlib import Path


def offered_skills(source: Path | None, target: str) -> dict[str, str]:
    """Each skill under `source`, by name, as `target/<name>` for the CLI.

    `source` is the directory as this process sees it; `target` is the same
    directory as the CLI will, which in a container is a mount path.
    """
    if source is None or not source.is_dir():
        return {}
    return {
        entry.name: f"{target.rstrip('/')}/{entry.name}"
        for entry in sorted(source.iterdir())
        if entry.is_dir() and (entry / "SKILL.md").is_file()
    }


def link_skills(directory: Path, *offers: Mapping[str, str]) -> None:
    """Make `directory` hold exactly one link per offered skill.

    A link per skill rather than one link for the directory, because both CLIs
    also treat this directory as their own: Codex installs `.system` skills
    into it on first run, and a directory link would put that write in the
    user's repository. Earlier offers win a name, so the workspace's own skill
    shadows a user's skill of the same name.
    """
    offered: dict[str, str] = {}
    for offer in offers:
        for name, target in offer.items():
            offered.setdefault(name, target)
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        directory.unlink()
    directory.mkdir(exist_ok=True)
    for existing in directory.iterdir():
        # Only links are this function's to remove. `.system` and anything
        # else a CLI owns is a real directory and stays.
        if existing.is_symlink() and offered.get(existing.name) != os.readlink(
            existing
        ):
            existing.unlink()
    for name, target in offered.items():
        link = directory / name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(target, target_is_directory=True)
