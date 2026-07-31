import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.codex_runtime.workspace import prepare_workspace
from backend.store import WorkspaceRegistry


class WorkspaceRuntimeTest(unittest.TestCase):
    def test_prepare_preserves_tracked_deletions_in_dirty_main(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            main_dir = temporary_root / "main"
            main_dir.mkdir()
            (main_dir / "present.txt").write_text("present\n", "utf-8")
            workspace_repo = temporary_root / "managed" / "repo"

            with (
                patch("backend.codex_runtime.workspace.MAIN_DIR", main_dir),
                patch(
                    "backend.codex_runtime.workspace.workspace_main_for",
                    return_value=workspace_repo,
                ),
                patch(
                    "backend.codex_runtime.workspace._tracked_entries",
                    return_value=(
                        [Path("present.txt"), Path("deleted.txt")],
                        [],
                    ),
                ),
            ):
                prepared = prepare_workspace("w_test")

            self.assertEqual(prepared, workspace_repo)
            self.assertEqual((prepared / "present.txt").read_text("utf-8"), "present\n")
            self.assertFalse((prepared / "deleted.txt").exists())
            subprocess.run(
                ["git", "-C", str(prepared), "rev-parse", "--verify", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )

    def test_failed_creation_cleanup_removes_descriptor_and_partial_repo(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            main_dir = temporary_root / "main"
            (main_dir / "logs").mkdir(parents=True)
            registry = WorkspaceRegistry(
                temporary_root / "agent-workspaces",
                main_dir=main_dir,
            )
            descriptor = registry.create("Failed", workspace_id="w_failed")
            partial_repo = registry.workspace_dir("w_failed") / "repo.tmp"
            partial_repo.mkdir()

            registry.discard_failed_creation(descriptor["workspace_id"])

            self.assertFalse(registry.workspace_dir("w_failed").exists())
            self.assertNotIn(
                "w_failed",
                {item["workspace_id"] for item in registry.list(include_archived=True)},
            )


if __name__ == "__main__":
    unittest.main()
