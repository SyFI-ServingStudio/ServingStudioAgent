"""Isolated Claude state with optional host-login credential supply."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ...runtime.invocation import InvocationHome, RoleContext
from ..skills import link_skills, offered_skills


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
        # `$CLAUDE_CONFIG_DIR/skills` is where Claude reads user-level skills,
        # and the config dir is this isolated home, so the user's own skills
        # appear only if they are linked here -- which happens on the host
        # alone. A container turn gets the workspace's and nothing else.
        user = (
            self.source / "skills"
            if self.source is not None and context.user_skills
            else None
        )
        link_skills(
            home.host / "skills",
            offered_skills(context.skills_source, context.skills),
            offered_skills(user, str(user)),
        )
