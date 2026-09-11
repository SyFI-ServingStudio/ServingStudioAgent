import asyncio
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx

from tests import test_application as fixtures
from tests.http_support import LiveRequest, sse_events
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.storage.database import Database


class ManagedJobTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ApplicationTests.setUp
    app = fixtures.ApplicationTests.app
    base = "/api/agent/v1"

    def providers(self, home, environment, prompts):
        configured = fixtures.ApplicationTests.providers(
            self, home, environment, prompts
        )
        owner = self

        class Adapter:
            adapter_id = "test"

            async def run(self, request):
                owner.request = request
                yield {"kind": "role_ready", "role": request.role.value}
                owner.entered.set()
                await owner.release.wait()
                yield {
                    "kind": "final",
                    "text": '{"action":"final_answer","message":"Done"}',
                }

        registry = ProviderRegistry()
        registry.register(
            replace(configured.registry.provider("test"), adapter=Adapter())
        )
        return replace(configured, registry=registry)

    async def asyncSetUp(self):
        Database.create(self.state / "workspace.sqlite")
        self.application = self.app()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.application), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        created = await self.client.post(
            self.base + "/workspaces/w_main/conversations", json={"agentMode": "single"}
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.cid = created.json()["id"]
        self.path = self.base + "/workspaces/w_main/conversations/" + self.cid
        await self.begin()

    async def begin(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.sending = asyncio.create_task(
            self.client.post(self.path + "/messages", json={"text": "work"})
        )
        self.addAsyncCleanup(self.finish)
        await asyncio.wait_for(self.entered.wait(), 3)
        context_path = self.application.state.managed_context.path("w_main", self.cid)
        self.token = json.loads(context_path.read_text())["capability_token"]
        self.headers = {"Authorization": "Bearer " + self.token}

    async def finish(self):
        self.release.set()
        return await asyncio.wait_for(self.sending, 3)

    async def post(self, route, payload, *, headers=None):
        return await self.client.post(
            self.base + "/internal/" + route,
            json=payload,
            headers=self.headers if headers is None else headers,
        )

    def events(self):
        return self.application.state.turns.storage("w_main").turns.events(
            self.cid, self.request.turn_id
        )

    async def simulation(self, root="/workspace/logs/experiment"):
        response = await self.post(
            "managed-runs/register",
            {"experimentRoot": root, "runCount": 2, "axes": ["tp"]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def artifact(self, kind="timing_predict", resource="p_prediction"):
        response = await self.post(
            "managed-jobs/register",
            {
                "jobKind": kind,
                "artifactRoot": "logs/" + resource,
                "analyzerResourceId": resource,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_simulation_rerun_preserves_identity_and_metadata_across_turns(self):
        first = await self.simulation()
        metadata_path = self.repo / "logs/experiment/experiment.meta.json"
        metadata = metadata_path.read_bytes()
        self.assertEqual(first["approvedRoot"], "/workspace/logs/experiment")
        self.assertEqual(json.loads(metadata)["experiment_id"], first["experimentId"])
        await self.finish()
        await self.begin()
        second = await self.simulation()
        self.assertEqual(first["experimentId"], second["experimentId"])
        self.assertNotEqual(first["jobId"], second["jobId"])
        self.assertEqual(metadata_path.read_bytes(), metadata)
        ready = await self.post(
            f"managed-runs/{second['jobId']}/status", {"status": "ready"}
        )
        self.assertEqual(ready.status_code, 200, ready.text)
        experiments = (await self.client.get(self.path + "/experiments")).json()[
            "experiments"
        ]
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0]["id"], first["experimentId"])
        self.assertEqual(experiments[0]["turn_id"], self.request.turn_id)
        self.assertEqual(experiments[0]["status"], "ready")

    async def test_callback_wakes_live_subscriber_before_provider_finishes(self):
        connection = LiveRequest(self.application, self.path + "/stream")
        self.addAsyncCleanup(connection.close)
        self.assertEqual((await connection.next())["type"], "http.response.start")
        while True:
            frame = await connection.next()
            events = sse_events(frame.get("body", b"").decode())
            if any(kind == "role_ready" for kind, _ in events):
                break
        job = await self.artifact()
        frame = await connection.next()
        self.assertEqual(sse_events(frame["body"].decode())[0][0], "job")
        self.assertEqual(
            sse_events(frame["body"].decode())[0][1]["jobId"], job["jobId"]
        )
        self.assertFalse(self.sending.done())
        self.assertFalse(self.release.is_set())

    async def test_three_typed_jobs_overlay_and_interleaved_browser_events(self):
        simulation = await self.simulation()
        artifacts = []
        for kind, resource in (
            ("timing_predict", "p_one"),
            ("kernel_profile", "kp_two"),
            ("kernel_measure", "km_three"),
        ):
            artifact = await self.artifact(kind, resource)
            self.assertEqual(artifact["approvedRoot"], "logs/" + resource)
            response = await self.post(
                f"managed-jobs/{artifact['jobId']}/status", {"status": "ready"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            overlay = await self.client.get(
                self.base + f"/workspaces/w_main/jobs/{artifact['resourceId']}"
            )
            self.assertEqual(overlay.status_code, 200)
            self.assertEqual(overlay.json()["analyzerResourceId"], resource)
            self.assertEqual(overlay.json()["status"], "ready")
            artifacts.append(artifact)
        response = await self.post(
            f"managed-runs/{simulation['jobId']}/status", {"status": "analysis_running"}
        )
        self.assertEqual(response.status_code, 200)
        listed = (await self.client.get(self.base + "/jobs")).json()["jobs"]
        self.assertEqual(
            {job["job_id"] for job in listed}, {job["jobId"] for job in artifacts}
        )
        self.assertTrue(
            all(
                "artifact_path" not in job
                and "descriptor" not in job
                and "summary" not in job
                for job in listed
            )
        )
        response = await self.finish()
        stream = [
            payload for kind, payload in sse_events(response.text) if kind == "job"
        ]
        self.assertEqual(
            [event["kind"] for event in stream],
            [
                "simulation.requested",
                "job.requested",
                "job.ready",
                "job.requested",
                "job.ready",
                "job.requested",
                "job.ready",
                "analysis.running",
            ],
        )
        replay = await self.client.get(
            self.path + f"/turns/{self.request.turn_id}/replay"
        )
        persisted = [
            payload
            for kind, payload in sse_events(replay.text)
            if kind in {event["kind"] for event in stream}
        ]
        self.assertEqual(persisted, stream)
        history = (await self.client.get(self.path)).json()
        activity = [
            item
            for item in history["messages"][-1]["activity"]
            if item["kind"] == "job"
        ]
        self.assertEqual(len(activity), len(stream))

    async def test_authority_wrong_type_and_invalid_resource_do_not_mutate(self):
        simulation = await self.simulation()
        artifact = await self.artifact()
        before = self.events()
        for route in (
            f"managed-jobs/{simulation['jobId']}/status",
            f"managed-runs/{artifact['jobId']}/status",
        ):
            self.assertEqual(
                (await self.post(route, {"status": "failed"})).status_code, 404
            )
        invalid = await self.post(
            "managed-jobs/register",
            {
                "jobKind": "kernel_profile",
                "artifactRoot": "logs/bad",
                "analyzerResourceId": "km_wrong",
            },
        )
        self.assertEqual(invalid.status_code, 400)
        missing = await self.post(
            "managed-runs/register", {"experimentRoot": "logs/denied"}, headers={}
        )
        self.assertEqual(missing.status_code, 401)
        wrong = self.application.state.capabilities.issue(
            workspace_id="w_main",
            conversation_id=self.cid,
            turn_id="another-turn",
            role="assistant",
        )
        response = await self.post(
            f"managed-runs/{simulation['jobId']}/status",
            {"status": "failed"},
            headers={"Authorization": "Bearer " + wrong.token},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.events(), before)
        await self.finish()
        expired = await self.post(
            f"managed-runs/{simulation['jobId']}/status", {"status": "failed"}
        )
        self.assertEqual(expired.status_code, 401)

    async def test_root_escape_and_metadata_conflict_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.repo / "logs").mkdir()
        (self.repo / "logs/escape").symlink_to(outside, target_is_directory=True)
        before = self.events()
        for root, status in (
            ("/workspace/logs", 400),
            ("../outside/run", 403),
            ("logs/escape/run", 403),
        ):
            response = await self.post(
                "managed-runs/register", {"experimentRoot": root}
            )
            self.assertEqual(response.status_code, status, response.text)
        path = self.repo / "logs/bad"
        path.mkdir()
        (path / "experiment.meta.json").symlink_to(outside / "secret")
        response = await self.post(
            "managed-runs/register", {"experimentRoot": "logs/bad"}
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.events(), before)
        self.assertEqual(list(outside.iterdir()), [])
        first = await self.simulation()
        metadata = self.repo / "logs/experiment/experiment.meta.json"
        metadata.write_text(
            json.dumps({"schema_version": 1, "experiment_id": "e_conflicting"})
        )
        before = self.events()
        response = await self.post(
            "managed-runs/register", {"experimentRoot": "logs/experiment"}
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.events(), before)
        self.assertEqual(
            self.application.state.jobs.storage("w_main").experiment_by_path(
                "experiment"
            )["id"],
            first["experimentId"],
        )

    async def test_metadata_retains_stable_identity_after_database_failure(self):
        from vibesim_agent.storage.jobs import Jobs

        before = self.events()
        capability = self.application.state.capabilities.authorize(self.token)
        with (
            patch.object(
                Jobs, "create_simulation", side_effect=OSError("database unavailable")
            ),
            self.assertRaisesRegex(OSError, "database unavailable"),
        ):
            self.application.state.jobs.register_run(
                capability, experiment_root="logs/retry", run_count=1, axes=[]
            )
        metadata = (self.repo / "logs/retry/experiment.meta.json").read_bytes()
        self.assertEqual(self.events(), before)
        await self.finish()
        await self.begin()
        registered = await self.simulation("logs/retry")
        self.assertEqual(
            registered["experimentId"], json.loads(metadata)["experiment_id"]
        )
        final = json.loads((self.repo / "logs/retry/experiment.meta.json").read_text())
        self.assertNotIn("agent_registration", final)
        self.assertEqual(final["origin"]["job_id"], registered["jobId"])
        self.assertEqual(final["origin"]["turn_id"], self.request.turn_id)
        self.assertIsNotNone(
            self.application.state.jobs.storage("w_main").get(final["origin"]["job_id"])
        )

    async def test_committed_job_origin_recovers_after_final_metadata_write_failure(
        self,
    ):
        service = self.application.state.jobs
        capability = self.application.state.capabilities.authorize(self.token)
        with (
            patch.object(
                service, "_finalize_metadata", side_effect=OSError("publish failed")
            ),
            self.assertRaisesRegex(OSError, "publish failed"),
        ):
            service.register_run(
                capability, experiment_root="logs/recover", run_count=1, axes=[]
            )
        path = self.repo / "logs/recover/experiment.meta.json"
        pending = json.loads(path.read_text())
        original_job_id = pending["agent_registration"]["origin"]["job_id"]
        self.assertIsNotNone(service.storage("w_main").get(original_job_id))
        registered = await self.simulation("logs/recover")
        final = json.loads(path.read_text())
        self.assertNotIn("agent_registration", final)
        self.assertEqual(final["origin"]["job_id"], original_job_id)
        self.assertNotEqual(registered["jobId"], original_job_id)

    async def test_unknown_pending_format_and_mismatched_provenance_are_rejected(self):
        first = await self.simulation()
        path = self.repo / "logs/experiment/experiment.meta.json"
        original = json.loads(path.read_text())
        malformed = [
            None,
            {},
            {"version": 2, "origin": original["origin"]},
            {"version": 1, "origin": {**original["origin"], "role": "implementer"}},
        ]
        before = self.events()
        for pending in malformed:
            with self.subTest(pending=pending):
                payload = {
                    **original,
                    "origin": {"kind": "managed"},
                    "agent_registration": pending,
                }
                path.write_text(json.dumps(payload))
                response = await self.post(
                    "managed-runs/register", {"experimentRoot": "logs/experiment"}
                )
                self.assertEqual(response.status_code, 409, response.text)
                self.assertEqual(json.loads(path.read_text()), payload)
                self.assertEqual(self.events(), before)
        other = self.repo / "logs/other"
        other.mkdir()
        payload = {
            **original,
            "origin": {"kind": "managed"},
            "agent_registration": {"version": 1, "origin": original["origin"]},
        }
        (other / "experiment.meta.json").write_text(json.dumps(payload))
        response = await self.post(
            "managed-runs/register", {"experimentRoot": "logs/other"}
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.events(), before)
        self.assertEqual(
            self.application.state.jobs.storage("w_main").get(first["jobId"])[
                "experiment_path"
            ],
            "experiment",
        )
