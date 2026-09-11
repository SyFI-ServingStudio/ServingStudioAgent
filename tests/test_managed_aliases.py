import unittest
from unittest.mock import patch

from tests import test_managed_jobs as fixtures


class ManagedAliasTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ManagedJobTests.setUp
    asyncSetUp = fixtures.ManagedJobTests.asyncSetUp
    providers = fixtures.ManagedJobTests.providers
    app = fixtures.ManagedJobTests.app
    begin = fixtures.ManagedJobTests.begin
    finish = fixtures.ManagedJobTests.finish
    events = fixtures.ManagedJobTests.events
    base = fixtures.ManagedJobTests.base
    prefixes = ("/api/agent/v1/internal/", "/api/internal/")

    def endpoints(self):
        return (
            ("managed-runs/register", {"experimentRoot": "logs/experiment"}),
            (
                "managed-jobs/register",
                {
                    "jobKind": "timing_predict",
                    "artifactRoot": "logs/p_test",
                    "analyzerResourceId": "p_test",
                },
            ),
            ("managed-runs/missing/status", {"status": "ready"}),
            ("managed-jobs/missing/status", {"status": "ready"}),
        )

    async def test_all_four_aliases_register_and_update_the_same_durable_jobs(self):
        jobs = self.application.state.jobs.storage("w_main")
        for index, prefix in enumerate(self.prefixes):
            run = await self.client.post(
                prefix + "managed-runs/register",
                headers=self.headers,
                json={"experimentRoot": f"logs/experiment-{index}"},
            )
            self.assertEqual(run.status_code, 200, run.text)
            run_id = run.json()["jobId"]
            artifact = await self.client.post(
                prefix + "managed-jobs/register",
                headers=self.headers,
                json={
                    "jobKind": "timing_predict",
                    "artifactRoot": f"logs/p_{index}",
                    "analyzerResourceId": f"p_{index}",
                },
            )
            self.assertEqual(artifact.status_code, 200, artifact.text)
            artifact_id = artifact.json()["jobId"]
            for family, job_id in (
                ("managed-runs", run_id),
                ("managed-jobs", artifact_id),
            ):
                for update_prefix in self.prefixes:
                    updated = await self.client.post(
                        update_prefix + f"{family}/{job_id}/status",
                        headers=self.headers,
                        json={"status": "ready"},
                    )
                    self.assertEqual(updated.status_code, 200, updated.text)
                    row = jobs.get(job_id)
                    self.assertEqual(
                        (row["conversation_id"], row["turn_id"], row["status"]),
                        (self.cid, self.request.turn_id, "ready"),
                    )
        events = self.events()
        self.assertEqual(
            sum(event["kind"] == "simulation.requested" for event in events), 2
        )
        self.assertEqual(sum(event["kind"] == "job.requested" for event in events), 2)
        self.assertEqual((await self.client.get("/api/jobs")).status_code, 404)

    async def test_aliases_share_exact_capability_auth_and_error_responses(self):
        before = self.events()
        for route, body in self.endpoints():
            for headers in ({}, {"Authorization": "Bearer invalid"}):
                responses = [
                    await self.client.post(prefix + route, headers=headers, json=body)
                    for prefix in self.prefixes
                ]
                self.assertEqual(
                    [response.status_code for response in responses], [401, 401]
                )
                self.assertEqual(responses[0].json(), responses[1].json())
            responses = [
                await self.client.post(prefix + route, headers=self.headers, json={})
                for prefix in self.prefixes
            ]
            self.assertEqual(
                [response.status_code for response in responses], [422, 422]
            )
            self.assertEqual(responses[0].json(), responses[1].json())
        for route in ("managed-runs/missing/status", "managed-jobs/missing/status"):
            responses = [
                await self.client.post(
                    prefix + route, headers=self.headers, json={"status": "ready"}
                )
                for prefix in self.prefixes
            ]
            self.assertEqual(
                [response.status_code for response in responses], [404, 404]
            )
            self.assertEqual(responses[0].json(), responses[1].json())
        self.assertEqual(self.events(), before)

    async def test_expired_capability_is_rejected_on_canonical_and_alias_without_writes(
        self,
    ):
        authority = self.application.state.capabilities
        capability = authority.authorize(self.token)
        before = self.events()
        with patch.object(authority, "clock", return_value=capability.expires_at + 1):
            for route, body in self.endpoints():
                for prefix in self.prefixes:
                    response = await self.client.post(
                        prefix + route, headers=self.headers, json=body
                    )
                    self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(self.events(), before)

    async def test_terminal_turn_with_valid_capability_conflicts_on_all_paths(self):
        await self.finish()
        capability = self.application.state.capabilities.issue(
            workspace_id="w_main",
            conversation_id=self.cid,
            turn_id=self.request.turn_id,
            role="assistant",
        )
        headers = {"Authorization": "Bearer " + capability.token}
        before = self.events()
        for route, body in self.endpoints():
            responses = [
                await self.client.post(prefix + route, headers=headers, json=body)
                for prefix in self.prefixes
            ]
            self.assertEqual(
                [response.status_code for response in responses], [409, 409]
            )
            self.assertEqual(responses[0].json(), responses[1].json())
        self.assertEqual(self.events(), before)
