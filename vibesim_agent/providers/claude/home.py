"""Isolated Claude state with optional host-login credential supply."""

import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ...runtime.invocation import InvocationHome, RoleContext


@dataclass(frozen=True)
class ClaudeProfile:
    source: Path | None = None

    def prepare(self, home: InvocationHome, context: RoleContext) -> None:
        if home.host.is_symlink():
            raise ValueError("Claude runtime home must not be a symlink")
        home.host.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.source is not None:
            source = self.source.resolve(strict=True)
            destination = home.host.resolve()
            if (
                destination == source
                or destination.is_relative_to(source)
                or source.is_relative_to(destination)
            ):
                raise ValueError(
                    "Claude credentials and runtime homes must be separate"
                )
            credential = source / ".credentials.json"
            target = home.host / ".credentials.json"
            marker = home.host / ".vibesim-auth-source-sha"
            if target.is_symlink() or marker.is_symlink():
                raise ValueError("Claude credential files must not be symlinks")
            content = credential.read_bytes()
            digest = sha256(content).hexdigest()
            # Preserve tokens refreshed by the CLI until the host login changes.
            previous = marker.read_text() if marker.is_file() else None
            if previous != digest or not target.is_file():
                target.touch(mode=0o600, exist_ok=True)
                target.chmod(0o600)
                target.write_bytes(content)
                marker.write_text(digest)
                marker.chmod(0o600)
        skills = home.host / "skills"
        # Re-point rather than create-if-absent: the role home outlives a switch
        # between execution modes, and a stale container path would otherwise
        # survive forever as a symlink into a directory that does not exist.
        if skills.is_symlink():
            if os.readlink(skills) != context.skills:
                skills.unlink()
                skills.symlink_to(context.skills, target_is_directory=True)
        elif not skills.exists():
            skills.symlink_to(context.skills, target_is_directory=True)
