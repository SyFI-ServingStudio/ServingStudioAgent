import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.codex_runtime import docker
from backend.codex_runtime.config import PROMPTS_DIR


class DockerAgentPromptMountTest(unittest.TestCase):
    def test_default_agent_prompt_uses_existing_workspace_mount_target(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_path = Path(temporary_directory)
            (workspace_path / "AGENTS.md").touch()

            self.assertEqual(
                docker._agent_prompt_mount_args(
                    workspace_path,
                    autonomous=False,
                ),
                [
                    "--mount",
                    (
                        f"type=bind,src={(PROMPTS_DIR / 'AGENTS.md').resolve()},"
                        "dst=/workspace/AGENTS.md,readonly"
                    ),
                ],
            )

    def test_autonomous_agent_prompt_uses_same_container_target(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_path = Path(temporary_directory)
            (workspace_path / "AGENTS.md").touch()

            self.assertEqual(
                docker._agent_prompt_mount_args(
                    workspace_path,
                    autonomous=True,
                ),
                [
                    "--mount",
                    (
                        "type=bind,"
                        f"src={(PROMPTS_DIR / 'AGENTS.autonomous.md').resolve()},"
                        "dst=/workspace/AGENTS.md,readonly"
                    ),
                ],
            )

    def test_agent_prompt_mount_refuses_to_create_workspace_target(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_path = Path(temporary_directory)

            with self.assertRaisesRegex(RuntimeError, "mount target is missing"):
                docker._agent_prompt_mount_args(
                    workspace_path,
                    autonomous=False,
                )

            self.assertFalse((workspace_path / "AGENTS.md").exists())


if __name__ == "__main__":
    unittest.main()
