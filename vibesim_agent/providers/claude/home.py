"""Isolated Claude state with optional host-login or AWS-profile credential supply."""

import configparser
import io
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ...runtime.invocation import InvocationHome, RoleContext
from ..skills import link_skills, offered_skills


_AWS_KEYS = ("aws_access_key_id", "aws_secret_access_key", "aws_session_token")


def _private_write(path: Path, content: str) -> None:
    if path.is_symlink():
        raise ValueError("Claude credential files must not be symlinks")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(content)
    path.chmod(0o600)


@dataclass(frozen=True)
class ClaudeProfile:
    source: Path | None = None
    # Bedrock profile mode: copy only this profile's static keys, never the
    # operator's other profiles or `~/.aws/config` (SSO caches and
    # credential_process helpers cannot run inside a runner anyway).
    aws_credentials: Path | None = None
    aws_profile: str | None = None

    def __post_init__(self):
        if (self.aws_credentials is None) != (self.aws_profile is None):
            raise ValueError("AWS credentials file and profile are set together")
        if self.aws_profile is not None and self.source is not None:
            raise ValueError("Claude login and AWS profile credentials are exclusive")

    def prepare(self, home: InvocationHome, context: RoleContext) -> None:
        if home.host.is_symlink():
            raise ValueError("Claude runtime home must not be a symlink")
        home.host.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.aws_profile is not None:
            self._copy_aws_profile(home.host / ".aws")
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

    def _copy_aws_profile(self, target: Path) -> None:
        assert self.aws_credentials is not None and self.aws_profile is not None
        source = configparser.RawConfigParser(interpolation=None)
        try:
            source.read_string(self.aws_credentials.read_text())
        except (OSError, UnicodeError, configparser.Error):
            raise ValueError("cannot read the AWS credentials file") from None
        if not source.has_section(self.aws_profile):
            raise ValueError("AWS profile is missing from the credentials file")
        section = source[self.aws_profile]
        keys = {key: section[key].strip() for key in _AWS_KEYS if key in section}
        if not keys.get("aws_access_key_id") or not keys.get("aws_secret_access_key"):
            raise ValueError("AWS profile has no static access key pair")
        if target.is_symlink():
            raise ValueError("Claude credential files must not be symlinks")
        target.mkdir(exist_ok=True, mode=0o700)
        copy = configparser.RawConfigParser(interpolation=None)
        copy[self.aws_profile] = keys
        rendered = io.StringIO()
        copy.write(rendered)
        # Rewritten on every prepare: no CLI-refreshed state lives here, and
        # a rotated host key reaches the next turn.
        _private_write(target / "credentials", rendered.getvalue())
        _private_write(target / "config", "")
