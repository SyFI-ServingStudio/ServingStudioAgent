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
    def test_analyzer_resource_ids_are_typed_by_job_kind(self) -> None:
        self.assertTrue(
            app_module._valid_analyzer_resource_id("timing_predict", "p_prediction")
        )
        self.assertTrue(
            app_module._valid_analyzer_resource_id("kernel_profile", "kp_profile")
        )
        self.assertTrue(
            app_module._valid_analyzer_resource_id("kernel_measure", "km_measure")
        )
        self.assertFalse(
            app_module._valid_analyzer_resource_id("kernel_profile", "km_measure")
        )
        self.assertFalse(
            app_module._valid_analyzer_resource_id("kernel_measure", None)
        )

    @staticmethod
    def _sweep_payload(experiment_id: str) -> dict:
        return {
            "protocol_version": 1,
            "schema_version": 1,
            "workspace_id": "root_0",
            "sweep_id": experiment_id,
            "axes": ["request_rate"],
            "metrics": [
                {
                    "key": "total_tps",
                    "label": "Total throughput",
                    "group": "throughput",
                    "unit": "token/s",
                    "objective": "maximize",
                }
            ],
            "runs": [
                {
                    "run_id": "r_test",
                    "coordinates": {"request_rate": 20},
                    "labels": {"request_rate": "20 req/s"},
                }
            ],
        }

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
                [
                    (experiment["id"], experiment["status"])
                    for experiment in experiments
                ],
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

    async def test_registers_typed_artifact_job_without_creating_an_experiment(
        self,
    ) -> None:
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
            artifact_root = (
                registry.repo_path("w_managed")
                / "logs"
                / "20260731_0_single_gemm_profile"
            )

            with patch.object(app_module, "store", store):
                registration = await app_module.register_managed_job(
                    app_module.RegisterManagedJob(
                        jobKind="kernel_profile",
                        artifactRoot="logs/20260731_0_single_gemm_profile",
                        analyzerResourceId="kp_profile_test",
                    ),
                    capability,
                )
                ready = await app_module.update_managed_job(
                    registration["jobId"],
                    app_module.UpdateManagedJob(
                        status="ready",
                    ),
                    capability,
                )

            self.assertFalse(artifact_root.exists())
            self.assertEqual(registration["resourceId"], ready["resourceId"])
            self.assertEqual(ready["jobKind"], "kernel_profile")
            self.assertNotIn("artifactPath", ready)
            self.assertNotIn("descriptor", ready)
            self.assertNotIn("summary", ready)
            with patch.object(app_module, "store", store):
                resource = app_module.get_managed_job_resource(
                    "w_managed",
                    registration["resourceId"],
                )
            self.assertEqual(resource["analyzerResourceId"], "kp_profile_test")
            self.assertNotIn("curve", resource)
            self.assertNotIn("files", resource)
            with patch.object(app_module, "store", store):
                catalog = app_module.list_managed_jobs()
            self.assertEqual(len(catalog["jobs"]), 1)
            self.assertEqual(catalog["jobs"][0]["workspace_id"], "w_managed")
            self.assertEqual(
                catalog["jobs"][0]["resource_id"], registration["resourceId"]
            )
            self.assertEqual(catalog["jobs"][0]["job_kind"], "kernel_profile")
            self.assertEqual(catalog["jobs"][0]["conversation_title"], "New chat")
            self.assertNotIn("descriptor", catalog["jobs"][0])
            self.assertNotIn("summary", catalog["jobs"][0])
            self.assertNotIn("artifact_path", catalog["jobs"][0])
            self.assertEqual(
                store.list_experiments(
                    "w_managed",
                    conversation_id="conversation",
                ),
                [],
            )
            self.assertEqual(
                [
                    event["kind"]
                    for event in store.list_turn_events("w_managed", "turn")
                ],
                ["job.requested", "job.ready"],
            )

    async def test_timing_job_links_to_analyzer_without_parsing_artifacts(self) -> None:
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
                registration = await app_module.register_managed_job(
                    app_module.RegisterManagedJob(
                        jobKind="timing_predict",
                        artifactRoot="logs/predict_llama",
                        analyzerResourceId="p_prediction_test",
                    ),
                    capability,
                )
                resource = app_module.get_managed_job_resource(
                    "w_managed",
                    registration["resourceId"],
                )

            self.assertEqual(
                registration["analyzerResourceId"], "p_prediction_test"
            )
            self.assertEqual(resource["analyzerResourceId"], "p_prediction_test")
            self.assertNotIn("files", resource)
            self.assertNotIn("iterBreakdown", resource)
            self.assertFalse(
                (registry.repo_path("w_managed") / "logs" / "predict_llama").exists()
            )

    async def test_registers_dynamic_citations_only_for_the_current_turn(self) -> None:
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
                role="orchestrator",
                expires_at=time.time() + 60,
            )

            with patch.object(app_module, "store", store):
                registration = await app_module.register_managed_run(
                    app_module.RegisterManagedRun(
                        experimentRoot="logs/20260731_citations",
                        runCount=1,
                        axes=["request_rate"],
                    ),
                    capability,
                )
                snapshot = app_module.register_managed_analyzer_citations(
                    app_module.RegisterManagedCitationDictionary(
                        experimentId=registration["experimentId"],
                        analysis=self._sweep_payload(registration["experimentId"]),
                    ),
                    capability,
                )

            self.assertEqual(snapshot["protocol"], "vibesim.citation-dictionary/v2")
            self.assertIn("`rate20`", snapshot["document"])
            self.assertTrue(
                any(
                    entry["token"] == "exp.rate20.throughput"
                    for entry in snapshot["entries"]
                )
            )
            events = store.list_turn_events(
                "w_managed", "turn", kinds={"citation.dictionary"}
            )
            self.assertEqual(len(events), 1)
            self.assertEqual(
                events[0]["payload"]["dictionary"]["entries"][0]["target"][
                    "workspaceId"
                ],
                "w_managed",
            )

            wrong_capability = Capability(
                token="wrong",
                workspace_id="w_managed",
                conversation_id="conversation",
                turn_id="other-turn",
                role="orchestrator",
                expires_at=time.time() + 60,
            )
            with (
                patch.object(app_module, "store", store),
                self.assertRaisesRegex(Exception, "not produced by this managed turn"),
            ):
                app_module.register_managed_analyzer_citations(
                    app_module.RegisterManagedCitationDictionary(
                        experimentId=registration["experimentId"],
                        analysis=self._sweep_payload(registration["experimentId"]),
                    ),
                    wrong_capability,
                )


if __name__ == "__main__":
    unittest.main()
