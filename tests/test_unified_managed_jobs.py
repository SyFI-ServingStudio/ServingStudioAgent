import unittest
from unittest.mock import patch

from tests import test_managed_jobs_v2 as fixtures
from vibesim_agent.storage.database import Database


class UnifiedManagedJobTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ManagedJobTests.setUp
    asyncSetUp = fixtures.ManagedJobTests.asyncSetUp
    providers = fixtures.ManagedJobTests.providers
    app = fixtures.ManagedJobTests.app
    begin = fixtures.ManagedJobTests.begin
    finish = fixtures.ManagedJobTests.finish
    events = fixtures.ManagedJobTests.events
    post = fixtures.ManagedJobTests.post
    base = fixtures.ManagedJobTests.base

    kinds = (
        ("simulation", None),
        ("timing_predict", "p_prediction"),
        ("kernel_profile", "kp_profile"),
        ("kernel_measure", "km_measure"),
    )

    def snapshot(self):
        with Database(self.state / "workspace.sqlite").connect() as connection:
            return tuple(connection.iterdump())

    def body(self, kind="simulation", resource=None, *, suffix=""):
        return {
            "job_kind": kind,
            "artifact_root": "logs/" + kind + suffix,
            "analyzer_resource_id": resource,
        }

    async def register(self, kind="simulation", resource=None, *, suffix=""):
        response = await self.post(
            "jobs/register", self.body(kind, resource, suffix=suffix)
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_four_kinds_register_and_update_with_durable_correct_events(self):
        jobs = self.application.state.jobs.storage("w_main")
        for kind, resource in self.kinds:
            registered = await self.register(kind, resource)
            job_id = registered["jobId"]
            row = jobs.get(job_id)
            self.assertEqual(row["job_kind"], kind)
            self.assertEqual(
                (row["conversation_id"], row["turn_id"]),
                (self.cid, self.request.turn_id),
            )
            requested = self.events()[-1]["payload"]
            if kind == "simulation":
                self.assertEqual(requested["kind"], "simulation.requested")
                self.assertEqual((requested["runCount"], requested["axes"]), (1, []))
                self.assertEqual(requested["experimentId"], registered["experimentId"])
            else:
                self.assertEqual(requested["kind"], "job.requested")
                self.assertEqual(requested["jobKind"], kind)
                self.assertEqual(requested["analyzerResourceId"], resource)
            updated = await self.post(f"jobs/{job_id}/status", {"status": "ready"})
            self.assertEqual(updated.status_code, 200, updated.text)
            self.assertEqual(jobs.get(job_id)["status"], "ready")
            self.assertEqual(updated.json(), self.events()[-1]["payload"])
            self.assertEqual(
                updated.json()["kind"],
                "experiment.ready" if kind == "simulation" else "job.ready",
            )

    async def test_camel_case_and_old_new_callbacks_interoperate_without_weakening_typed_routes(
        self,
    ):
        for kind, resource in self.kinds:
            family = "managed-runs" if kind == "simulation" else "managed-jobs"
            new_body = {
                "jobKind": kind,
                "artifactRoot": "logs/new-" + kind,
                "analyzerResourceId": resource,
                "runCount": 3 if kind == "simulation" else 1,
                "axes": ["tp"] if kind == "simulation" else [],
            }
            new = await self.post("jobs/register", new_body)
            self.assertEqual(new.status_code, 200, new.text)
            if kind == "simulation":
                self.assertEqual(self.events()[-1]["payload"]["runCount"], 3)
                self.assertEqual(self.events()[-1]["payload"]["axes"], ["tp"])
            job_id = new.json()["jobId"]
            old_update = await self.post(
                f"{family}/{job_id}/status", {"status": "running"}
            )
            self.assertEqual(old_update.status_code, 200, old_update.text)
            wrong_family = "managed-jobs" if kind == "simulation" else "managed-runs"
            before = self.snapshot()
            wrong = await self.post(
                f"{wrong_family}/{job_id}/status", {"status": "failed"}
            )
            self.assertEqual(wrong.status_code, 404, wrong.text)
            self.assertEqual(self.snapshot(), before)
            old_body = (
                {"experimentRoot": "logs/old-simulation"}
                if kind == "simulation"
                else {
                    "jobKind": kind,
                    "artifactRoot": "logs/old-" + kind,
                    "analyzerResourceId": resource + "_old",
                }
            )
            old = await self.post(f"{family}/register", old_body)
            self.assertEqual(old.status_code, 200, old.text)
            updated = await self.post(
                f"jobs/{old.json()['jobId']}/status", {"status": "ready"}
            )
            self.assertEqual(updated.status_code, 200, updated.text)

    async def test_invalid_kind_fields_and_status_do_not_write(self):
        before = self.snapshot()
        for body, expected in (
            ({**self.body(), "job_kind": "unknown"}, 422),
            ({**self.body(), "run_count": 0}, 422),
            ({"job_kind": "simulation"}, 422),
            (self.body("kernel_profile", "km_wrong"), 400),
            (self.body("simulation", "p_wrong"), 400),
            ({**self.body("timing_predict", "p_test"), "run_count": 2}, 400),
            ({**self.body("timing_predict", "p_test"), "axes": ["tp"]}, 400),
        ):
            response = await self.post("jobs/register", body)
            self.assertEqual(response.status_code, expected, response.text)
            self.assertEqual(self.snapshot(), before)
        job = await self.register()
        before = self.snapshot()
        for job_id, status, expected in (
            (job["jobId"], "invalid", 400),
            ("missing", "ready", 404),
        ):
            response = await self.post(f"jobs/{job_id}/status", {"status": status})
            self.assertEqual(response.status_code, expected, response.text)
            self.assertEqual(self.snapshot(), before)

    async def test_missing_expired_and_foreign_capabilities_cannot_mutate(self):
        job = await self.register()
        routes = (
            ("jobs/register", self.body(suffix="-denied")),
            (f"jobs/{job['jobId']}/status", {"status": "failed"}),
        )
        before = self.snapshot()
        for route, body in routes:
            for headers in ({}, {"Authorization": "Bearer invalid"}):
                response = await self.post(route, body, headers=headers)
                self.assertEqual(response.status_code, 401, response.text)
        authority = self.application.state.capabilities
        capability = authority.authorize(self.token)
        with patch.object(authority, "clock", return_value=capability.expires_at + 1):
            for route, body in routes:
                response = await self.post(route, body)
                self.assertEqual(response.status_code, 401, response.text)
        for fields in (
            {"turn_id": "another-turn"},
            {"conversation_id": "another-conversation"},
            {"workspace_id": "w_missing"},
        ):
            identity = {
                "workspace_id": "w_main",
                "conversation_id": self.cid,
                "turn_id": self.request.turn_id,
                "role": "assistant",
                **fields,
            }
            wrong = authority.issue(**identity)
            for route, body in routes:
                response = await self.post(
                    route, body, headers={"Authorization": "Bearer " + wrong.token}
                )
                self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(self.snapshot(), before)

    async def test_existing_running_foreign_owner_and_later_turn_cannot_update_job(
        self,
    ):
        job = await self.register()
        created = await self.client.post(
            self.base + "/workspaces/w_main/conversations", json={"agentMode": "single"}
        )
        self.assertEqual(created.status_code, 200, created.text)
        other_cid = created.json()["id"]
        turns = self.application.state.turns.storage("w_main").turns
        turns.start(other_cid, "other-running-turn", "independent owner")
        self.assertEqual(
            turns.get(other_cid, "other-running-turn")["status"], "running"
        )
        other = self.application.state.capabilities.issue(
            workspace_id="w_main",
            conversation_id=other_cid,
            turn_id="other-running-turn",
            role="assistant",
        )
        before = self.snapshot()
        response = await self.post(
            f"jobs/{job['jobId']}/status",
            {"status": "failed"},
            headers={"Authorization": "Bearer " + other.token},
        )
        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(self.snapshot(), before)
        await self.finish()
        await self.begin()
        before = self.snapshot()
        response = await self.post(f"jobs/{job['jobId']}/status", {"status": "failed"})
        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(self.snapshot(), before)

    async def test_terminal_turn_with_valid_capability_rejects_both_endpoints(self):
        job = await self.register()
        await self.finish()
        token = self.application.state.capabilities.issue(
            workspace_id="w_main",
            conversation_id=self.cid,
            turn_id=self.request.turn_id,
            role="assistant",
        ).token
        before = self.snapshot()
        for route, body in (
            ("jobs/register", self.body(suffix="-late")),
            (f"jobs/{job['jobId']}/status", {"status": "failed"}),
        ):
            response = await self.post(
                route, body, headers={"Authorization": "Bearer " + token}
            )
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(self.snapshot(), before)
