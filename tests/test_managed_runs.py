from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend import app as app_module
from backend.managed_context import Capability
from backend.store import Store, WorkspaceRegistry


class ManagedRunApiTests(unittest.IsolatedAsyncioTestCase):
    def test_agent_can_create_a_workspace_before_starting_conversations(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            main_dir = root / "main"
            (main_dir / "logs").mkdir(parents=True)
            registry = WorkspaceRegistry(root / "agent-workspaces", main_dir=main_dir)
            store = Store(registry)

            with (
                patch.object(app_module, "store", store),
                patch.object(
                    app_module,
                    "prepare_workspace",
                    return_value=registry.workspace_dir("w_placeholder") / "repo",
                ) as prepare_workspace,
            ):
                descriptor = app_module.agent_create_workspace(
                    app_module.NewWorkspace(displayName="Agent study"),
                    None,
                )

            self.assertEqual(descriptor["display_name"], "Agent study")
            self.assertEqual(descriptor["storage_kind"], "managed")
            prepare_workspace.assert_called_once_with(descriptor["workspace_id"])

    async def test_registers_status_and_stable_analyzer_identity(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            main_dir = root / "main"
            (main_dir / "logs").mkdir(parents=True)
            registry = WorkspaceRegistry(root / "agent-workspaces", main_dir=main_dir)
            registry.create("Managed", workspace_id="w_managed")
            (registry.repo_path("w_managed") / "logs").mkdir(parents=True)
            store = Store(registry)
            store.create("w_managed", "conversation", "workspace-write")
            store.start_turn("w_managed", "conversation", "turn")
            capability = Capability(
                token="token",
                workspace_id="w_managed",
                conversation_id="conversation",
                turn_id="turn",
                role="implementer",
                expires_at=time.time() + 60,
            )

            with patch.object(app_module, "store", store):
                registration = await app_module.register_managed_run(
                    app_module.RegisterManagedRun(
                        experimentRoot="logs/20260728_test",
                        runCount=4,
                        axes=["request_rate", "tensor_parallel"],
                    ),
                    capability,
                )
                ready = await app_module.update_managed_run(
                    registration["jobId"],
                    app_module.UpdateManagedRun(status="ready"),
                    capability,
                )

            metadata_path = (
                registry.logs_path("w_managed")
                / "20260728_test"
                / "experiment.meta.json"
            )
            metadata = json.loads(metadata_path.read_text("utf-8"))
            self.assertEqual(metadata["experiment_id"], registration["experimentId"])
            self.assertEqual(metadata["origin"]["workspace_id"], "w_managed")
            self.assertEqual(metadata["origin"]["conversation_id"], "conversation")
            self.assertEqual(ready["status"], "ready")
            experiments = store.list_experiments(
                "w_managed",
                conversation_id="conversation",
            )
            self.assertEqual(
                [(experiment["id"], experiment["status"]) for experiment in experiments],
                [(registration["experimentId"], "ready")],
            )
            self.assertEqual(
                [
                    event["kind"]
                    for event in store.list_turn_events("w_managed", "turn")
                ],
                ["simulation.requested", "experiment.ready"],
            )
            immutable_metadata = metadata_path.read_bytes()
            with patch.object(app_module, "store", store):
                rerun = await app_module.register_managed_run(
                    app_module.RegisterManagedRun(
                        experimentRoot="logs/20260728_test",
                        runCount=4,
                        axes=["request_rate", "tensor_parallel"],
                    ),
                    capability,
                )
            self.assertEqual(rerun["experimentId"], registration["experimentId"])
            self.assertNotEqual(rerun["jobId"], registration["jobId"])
            self.assertEqual(metadata_path.read_bytes(), immutable_metadata)


if __name__ == "__main__":
    unittest.main()
