from pathlib import Path

from backend.codex_runtime import docker
from backend.codex_runtime.config import PROMPTS_DIR


def test_default_agent_prompt_is_mounted_at_workspace_contract_path() -> None:
    assert docker._agent_prompt_mount_args(autonomous=False) == [
        "-v",
        f"{(PROMPTS_DIR / 'AGENTS.md').resolve()}:/workspace/AGENTS.md:ro",
    ]


def test_autonomous_agent_prompt_uses_same_container_target() -> None:
    mount_arguments = docker._agent_prompt_mount_args(autonomous=True)

    assert mount_arguments == [
        "-v",
        f"{(PROMPTS_DIR / 'AGENTS.autonomous.md').resolve()}:/workspace/AGENTS.md:ro",
    ]
    assert Path(mount_arguments[1].split(":", 1)[0]).is_file()
