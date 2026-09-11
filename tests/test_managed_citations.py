import asyncio
import json
import unittest
from dataclasses import replace
from unittest.mock import patch
from urllib.error import HTTPError

from tests import test_application as application_fixture
from tests import test_managed_jobs as job_fixture
from tests.http_support import sse_events
from tests.test_analyzer_evidence_mcp import _JsonResponse
from vibesim_agent.analyzer_evidence_mcp import server
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

    async def test_modern_mcp_resources_register_through_http_and_preserve_targets(
        self,
    ):
        loop = asyncio.get_running_loop()
        payload = {"series": [{"metric": "time_ms"}], "runtime_ms": {"median": 1.2}}
        registrations = []

        def open_request(request, timeout):
            if request.method == "GET":
                return _JsonResponse(payload)
            response = asyncio.run_coroutine_threadsafe(
                self.client.post(
                    request.full_url,
                    content=request.data,
                    headers=dict(request.header_items()),
                ),
                loop,
            ).result(timeout=3)
            if response.status_code != 200:
                raise HTTPError(
                    request.full_url,
                    response.status_code,
                    response.text,
                    {},
                    _JsonResponse(response.json()),
                )
            dictionary = response.json()
            registrations.append(dictionary)
            return _JsonResponse(dictionary)

        paths = [
            ("predictions/p_test/descriptor", None),
            ("predictions/p_test/subjects/cases/payload", None),
            (
                "predictions/p_test/cases/0/subjects/optimality-waterfall/payload?mode=batch_locked",
                "optimality-breakdown",
            ),
            (
                "predictions/p_test/cases/0/subjects/optimality-kernel-ladder/payload",
                "optimality-kernel-ladder",
            ),
            (
                "predictions/p_test/cases/0/operations/2/subjects/cost-tree/payload",
                "cost-tree",
            ),
            (
                "predictions/p_test/cases/0/operations/2/leaves/3/subjects/kernel-throughput-analysis/payload",
                "kernel-throughput",
            ),
            (
                "predictions/p_test/subjects/kernel-input-distribution/payload",
                "kernel-input-distribution",
            ),
            ("runs/r_test/subjects/utilization/payload", "utilization"),
            (
                "runs/r_test/workers/prefill/0/operations/1/2/3/subjects/cost-tree/payload",
                "cost-tree",
            ),
            (
                "runs/r_test/workers/prefill/0/operations/1/2/3/leaves/4/subjects/kernel-throughput-analysis/payload",
                "kernel-throughput",
            ),
            ("kernel-profiles/kp_test/subjects/curve/payload", "curve"),
            ("kernel-measurements/km_test/subjects/summary/report", "summary"),
        ]
        with (
            patch.object(
                server,
                "_managed_context",
                return_value={
                    "backend_url": "http://test",
                    "capability_token": self.token,
                },
            ),
            patch.object(server, "_base_url", return_value="http://analyzer.test"),
            patch.object(server, "urlopen", open_request),
        ):
            for path, panel in paths:
                with self.subTest(path=path):
                    result = await asyncio.to_thread(
                        server.read_analyzer_resource, "/api/analyzer/v1/" + path
                    )
                    self.assertEqual(result["result"], payload)
                    if (
                        path
                        == "predictions/p_test/cases/0/operations/2/subjects/cost-tree/payload"
                    ):
                        self.final_text = f"See `{result['citation']}`."
                    tokens = (
                        [result["citation"]]
                        if "citation" in result
                        else list(result["citations"].values())
                    )
                    entries = {
                        entry["token"]: entry for entry in registrations[-1]["entries"]
                    }
                    for token in tokens:
                        target = entries[token]["target"]
                        self.assertEqual(target["workspaceId"], "w_main")
                        self.assertEqual(target["panelId"], panel)
                        if "/leaves/4/" in path:
                            self.assertEqual(target["leafId"], 4)
                            self.assertEqual(
                                target["operation"],
                                {"iterId": "1", "batchId": "2", "operationId": "3"},
                            )
        self.assertEqual(len(registrations), len(paths))
        await self.finish()
        history = (await self.client.get(self.path)).json()["messages"][-1]
        self.assertEqual(history["citations"][0]["target"]["operationId"], "2")

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
        self.assertEqual(
            snapshots[-1]["payload"]["dictionary"],
            {
                key: value
                for key, value in dictionary.items()
                if key != "registeredEntries"
            },
        )
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
