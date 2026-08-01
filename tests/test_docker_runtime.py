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


class DockerLegacyRuntimeMigrationTest(unittest.TestCase):
    def test_imports_shared_sessions_without_removing_legacy_state(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            legacy_home = Path(temporary_directory) / "conversation"
            role_home = legacy_home / "orchestrator"
            legacy_session = legacy_home / "sessions" / "2026" / "rollout.jsonl"
            legacy_snapshot = legacy_home / "shell_snapshots" / "session.sh"
            legacy_session.parent.mkdir(parents=True)
            legacy_snapshot.parent.mkdir(parents=True)
            legacy_session.write_text("legacy session", encoding="utf-8")
            legacy_snapshot.write_text("legacy snapshot", encoding="utf-8")
            role_home.mkdir()

            docker._import_legacy_shared_runtime(legacy_home, role_home)

            self.assertEqual(
                (role_home / "sessions" / "2026" / "rollout.jsonl").read_text(
                    encoding="utf-8"
                ),
                "legacy session",
            )
            self.assertEqual(
                (role_home / "shell_snapshots" / "session.sh").read_text(
                    encoding="utf-8"
                ),
                "legacy snapshot",
            )
            self.assertTrue(legacy_session.exists())
            self.assertTrue(legacy_snapshot.exists())

    def test_import_is_idempotent_after_role_runtime_changes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            legacy_home = Path(temporary_directory) / "conversation"
            role_home = legacy_home / "implementer"
            legacy_session = legacy_home / "sessions" / "rollout.jsonl"
            role_session = role_home / "sessions" / "rollout.jsonl"
            legacy_session.parent.mkdir(parents=True)
            legacy_session.write_text("legacy", encoding="utf-8")
            role_home.mkdir()

            docker._import_legacy_shared_runtime(legacy_home, role_home)
            role_session.write_text("role-specific", encoding="utf-8")
            docker._import_legacy_shared_runtime(legacy_home, role_home)

            self.assertEqual(role_session.read_text(encoding="utf-8"), "role-specific")


if __name__ == "__main__":
    unittest.main()
