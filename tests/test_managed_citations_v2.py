import json
import unittest
from dataclasses import replace

from tests import test_application as application_fixture
from tests import test_managed_jobs_v2 as job_fixture
from tests.http_support import sse_events
from vibesim_agent.providers.registry import ProviderRegistry


class ManagedCitationTests(unittest.IsolatedAsyncioTestCase):
    setUp = job_fixture.ManagedJobTests.setUp
    asyncSetUp = job_fixture.ManagedJobTests.asyncSetUp
    app = job_fixture.ManagedJobTests.app
    begin = job_fixture.ManagedJobTests.begin
    finish = job_fixture.ManagedJobTests.finish
    post = job_fixture.ManagedJobTests.post
    events = job_fixture.ManagedJobTests.events
    simulation = job_fixture.ManagedJobTests.simulation
    base = job_fixture.ManagedJobTests.base

    def providers(self, home, environment, prompts):
        configured = application_fixture.ApplicationTests.providers(
            self, home, environment, prompts
        )
        self.final_text = "Done"
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
                    "text": json.dumps(
                        {
                            "action": "final_answer",
                            "message": owner.final_text,
                        }
                    ),
                }

        registry = ProviderRegistry()
        registry.register(
            replace(configured.registry.provider("test"), adapter=Adapter())
        )
        return replace(configured, registry=registry)

    @staticmethod
    def aggregate(experiment_id="e_external"):
        return {
            "resourceKind": "aggregate",
            "experimentId": experiment_id,
            "analysis": {
                "protocol_version": 1,
                "schema_version": 1,
                "workspace_id": "another-workspace",
                "sweep_id": experiment_id,
                "axes": [],
                "runs": [],
                "metrics": [
                    {
                        "key": "total_tps",
                        "label": "Total throughput",
                        "group": "throughput",
                        "unit": "token/s",
                        "objective": "maximize",
                    }
                ],
            },
        }

    @staticmethod
    def run_resource(run_id="r_test"):
        return {
            "resourceKind": "run",
            "runId": run_id,
            "resourcePath": f"/api/v1/runs/{run_id}/subjects/utilization/payload",
        }

    async def register(self, payload):
        response = await self.post("analyzer-citations/register", payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_five_resource_kinds_freeze_dynamic_tokens_in_history_done_and_replay(
        self,
    ):
        requests = [
            self.aggregate(),
            self.run_resource(),
            {
                "resourceKind": "prediction",
                "predictionId": "p_test",
                "resourcePath": "/api/v1/predictions/p_test/cases/40/optimality-waterfall?mode=batch_locked",
            },
            {
                "resourceKind": "kernel_profile",
                "profileId": "kp_test",
                "resourcePath": "/api/v1/kernel-profiles/kp_test/curve",
                "analysis": {"series": [{"metric": "time_ms"}, {"metric": "tflops"}]},
            },
            {
                "resourceKind": "kernel_measurement",
                "measurementId": "km_test",
                "resourcePath": "/api/v1/kernel-measurements/km_test/summary",
                "analysis": {"runtime_ms": {"median": 1.2, "p99": 1.8}},
            },
        ]
        for body in requests:
            dictionary = await self.register(body)
        entries = {entry["token"]: entry for entry in dictionary["entries"]}
        self.assertEqual(
            set(entries),
            {
                "exp.throughput",
                "run.cluster.utilization",
                "pred.casev40.batch_locked.optimality-breakdown",
                "kprof.curve.time_ms",
                "kprof.curve.tflops",
                "kmeasure.summary.median",
                "kmeasure.summary.p99",
            },
        )
        target = entries["run.cluster.utilization"]["target"]
        self.assertIn("poolRole", target)
        self.assertIsNone(target["poolRole"])
        snapshots = [
            event for event in self.events() if event["kind"] == "citation.dictionary"
        ]
        self.assertEqual(len(snapshots), 5)
        self.assertEqual(snapshots[-1]["payload"]["dictionary"], dictionary)
        self.final_text = (
            "Evidence " + " ".join(f"`{token}`" for token in entries) + "."
        )
        response = await self.finish()
        done = [
            payload for kind, payload in sse_events(response.text) if kind == "done"
        ]
        self.assertEqual(len(done), 1)
        history = (await self.client.get(self.path)).json()["messages"][-1]
        citations = history["citations"]
        self.assertEqual({citation["token"] for citation in citations}, set(entries))
        self.assertEqual(done[0]["citations"], citations)
        for citation in citations:
            self.assertEqual(citation["target"]["workspaceId"], "w_main")
            self.assertEqual(
                self.final_text[citation["sourceStart"] : citation["sourceEnd"]],
                "`" + citation["token"] + "`",
            )
        replay = await self.client.get(
            self.path + f"/turns/{self.request.turn_id}/replay"
        )
        replayed = sse_events(replay.text)
        self.assertEqual(
            [payload["citations"] for kind, payload in replayed if kind == "done"],
            [citations],
        )
        self.assertEqual(
            [payload for kind, payload in replayed if kind == "citation.dictionary"],
            [event["payload"] for event in snapshots],
        )

    async def test_same_turn_merge_keeps_other_tokens_and_freezes_latest_target(self):
        await self.register(self.aggregate())
        await self.register(self.run_resource("r_old"))
        latest = await self.register(self.run_resource("r_new"))
        entries = {entry["token"]: entry for entry in latest["entries"]}
        self.assertEqual(set(entries), {"exp.throughput", "run.cluster.utilization"})
        self.assertEqual(entries["run.cluster.utilization"]["target"]["runId"], "r_new")
        self.assertIsNone(entries["run.cluster.utilization"]["target"]["poolRole"])
        self.final_text = "Latest `run.cluster.utilization`."
        await self.finish()
        history = (await self.client.get(self.path)).json()["messages"][-1]
        self.assertEqual(history["citations"][0]["target"]["runId"], "r_new")

    async def test_external_aggregate_without_current_job_is_allowed(self):
        body = self.aggregate("e_other_workspace")
        jobs = self.application.state.jobs.storage("w_main")
        self.assertIsNone(jobs.get_experiment("e_other_workspace"))
        dictionary = await self.register(body)
        self.assertEqual(
            dictionary["entries"][0]["target"]["experimentId"], "e_other_workspace"
        )
        self.assertEqual(dictionary["entries"][0]["target"]["workspaceId"], "w_main")
        self.assertIsNone(jobs.get_experiment("e_other_workspace"))

    async def test_known_nonready_experiment_conflicts_without_dictionary_event(self):
        registered = await self.simulation()
        before = self.events()
        response = await self.post(
            "analyzer-citations/register", self.aggregate(registered["experimentId"])
        )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.events(), before)
        status = await self.post(
            f"managed-runs/{registered['jobId']}/status", {"status": "ready"}
        )
        self.assertEqual(status.status_code, 200, status.text)
        await self.register(self.aggregate(registered["experimentId"]))

    async def test_invalid_parameters_and_capability_never_append_dictionary(self):
        before = self.events()
        for kind in (
            "aggregate",
            "run",
            "prediction",
            "kernel_profile",
            "kernel_measurement",
        ):
            with self.subTest(kind=kind):
                response = await self.post(
                    "analyzer-citations/register", {"resourceKind": kind}
                )
                self.assertEqual(response.status_code, 400, response.text)
        response = await self.post(
            "analyzer-citations/register", {"resourceKind": "unsupported"}
        )
        self.assertEqual(response.status_code, 422)
        for headers in ({}, {"Authorization": "Bearer invalid"}):
            response = await self.post(
                "analyzer-citations/register", self.run_resource(), headers=headers
            )
            self.assertEqual(response.status_code, 401)
        self.assertEqual(self.events(), before)

    async def test_terminal_turn_rejects_even_still_valid_capability_without_event(
        self,
    ):
        await self.finish()
        before = self.events()
        capability = self.application.state.capabilities.issue(
            workspace_id="w_main",
            conversation_id=self.cid,
            turn_id=self.request.turn_id,
            role="assistant",
        )
        self.assertIsNotNone(
            self.application.state.capabilities.authorize(capability.token)
        )
        response = await self.post(
            "analyzer-citations/register",
            self.run_resource(),
            headers={"Authorization": "Bearer " + capability.token},
        )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.events(), before)
