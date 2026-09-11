"""Claude keeps isolated state and receives credentials through its environment."""

from dataclasses import dataclass

from ...runtime.invocation import InvocationHome


@dataclass(frozen=True)
class ClaudeProfile:
    def prepare(self, home: InvocationHome) -> None:
        if home.host.is_symlink():
            raise ValueError("Claude runtime home must not be a symlink")
        home.host.mkdir(parents=True, exist_ok=True, mode=0o700)
        skills = home.host / "skills"
        if not skills.is_symlink() and not skills.exists():
            skills.symlink_to("/workspace/skills", target_is_directory=True)
