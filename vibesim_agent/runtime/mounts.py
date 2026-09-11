"""Explicit bind mounts; validation never creates host paths or runs Docker."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..settings import ContainerSettings

WORKSPACE_TARGET = PurePosixPath("/workspace")
PROMPTS_TARGET = PurePosixPath("/opt/vibesim/prompts")
MCP_TARGET = PurePosixPath("/opt/vibesim/analyzer-evidence-mcp")
HF_TARGET = PurePosixPath("/model")


def managed_context_target(
    value: str, settings: ContainerSettings, role_root: str
) -> PurePosixPath:
    """Reserve a dedicated read-only directory for replaceable capability files."""
    _mount_path(value)
    path = PurePosixPath(value)
    directory = path.parent
    if (
        path.anchor != "/"
        or ".." in path.parts
        or directory == PurePosixPath("/")
        or value.endswith("/")
    ):
        raise ValueError(
            "managed context must be a file in an absolute dedicated directory"
        )
    reserved = (
        WORKSPACE_TARGET,
        PROMPTS_TARGET,
        MCP_TARGET,
        HF_TARGET,
        PurePosixPath("/candidate"),
        PurePosixPath(settings.home),
        PurePosixPath(settings.uv_project_environment),
        PurePosixPath(settings.uv_cache_dir),
        PurePosixPath(role_root),
    )
    if any(
        directory.is_relative_to(target) or target.is_relative_to(directory)
        for target in reserved
    ):
        raise ValueError("managed context directory overlaps a runtime path")
    return path


def _mount_path(value: str) -> None:
    # Docker parses each --mount argument as CSV. Reject ambiguous field syntax
    # instead of allowing a host path to inject another mount option.
    if any(
        character in ',"' or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise ValueError(
            "mount paths cannot contain commas, quotes or control characters"
        )


@dataclass(frozen=True)
class Mount:
    source: Path
    target: PurePosixPath
    read_only: bool = False

    def __post_init__(self):
        source = Path(self.source)
        target = PurePosixPath(self.target)
        _mount_path(str(source))
        _mount_path(str(target))
        if not source.is_absolute():
            raise ValueError("mount source must be absolute")
        if target.anchor != "/" or target == PurePosixPath("/") or ".." in target.parts:
            raise ValueError(
                "mount target must be an absolute non-root path without traversal"
            )
        if not source.exists():
            raise ValueError(f"mount source does not exist: {source}")
        source = source.resolve(strict=True)
        _mount_path(str(source))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)

    def docker_args(self) -> list[str]:
        fields = ["type=bind", f"src={self.source}", f"dst={self.target}"]
        if self.read_only:
            fields.append("readonly")
        return ["--mount", ",".join(fields)]


def _directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{label} directory must exist at an absolute path: {path}")
    return path.resolve(strict=True)


def workspace_mounts(
    *,
    workspace: Path,
    main: Path,
    submodules: Sequence[Path],
    prompts: Path,
    agent_prompt: Path,
    mcp: Path,
    settings: ContainerSettings,
    peer_workspace: Path | None = None,
) -> tuple[Mount, ...]:
    """Compose workspace bindings; provider homes are separate explicit mounts.

    Host paths are already resolved by composition, including any user-home
    expansion. The container owner sets HF_HOME to HF_TARGET when configured.
    Submodules are relative gitlink paths discovered by the workspace owner.
    """
    workspace = _directory(workspace, "workspace")
    main = _directory(main, "main")
    prompts = _directory(prompts, "prompts")
    mcp = _directory(mcp, "Analyzer MCP")
    if not agent_prompt.is_absolute() or not agent_prompt.is_file():
        raise ValueError(
            f"agent prompt must be an existing absolute file: {agent_prompt}"
        )
    target = workspace / "AGENTS.md"
    if target.is_symlink() or not target.is_file():
        raise ValueError(
            "workspace AGENTS.md must be an existing regular mount target; "
            f"refusing to modify shared workspace: {target}"
        )
    mounts = [
        Mount(workspace, WORKSPACE_TARGET),
        Mount(mcp, MCP_TARGET, read_only=True),
        Mount(prompts, PROMPTS_TARGET, read_only=True),
        Mount(agent_prompt, WORKSPACE_TARGET / "AGENTS.md", read_only=True),
    ]
    seen = set()
    for path in submodules:
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative.parts[0] == "AGENTS.md"
            or relative in seen
        ):
            raise ValueError(f"invalid or duplicate submodule path: {path}")
        seen.add(relative)
        source = _directory(main / relative, "submodule")
        if not source.is_relative_to(main):
            raise ValueError(f"submodule source escapes main checkout: {path}")
        mounts.append(Mount(source, WORKSPACE_TARGET / relative, read_only=True))
    if peer_workspace is not None and peer_workspace.is_dir():
        mounts.append(
            Mount(peer_workspace, PurePosixPath("/candidate"), read_only=True)
        )
    if settings.hf_home is not None:
        source = _directory(settings.hf_home, "configured HF_HOME")
        mounts.append(Mount(source, HF_TARGET, read_only=True))
    return tuple(mounts)
