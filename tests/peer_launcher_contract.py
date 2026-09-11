"""Explicit cross-repository check using real Launcher clients and Agent HTTP.

Run with --launcher-root and --legacy-launcher-root pointing at the two source
checkouts. Only urllib's socket transport is bridged to the test ASGI app;
context reading, payload generation, validation, SQLite and events are real.
"""

import argparse
import asyncio
import importlib
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from tests import test_managed_jobs_v2 as fixtures


def load_clients(root: Path, package_name: str):
    package = types.ModuleType(package_name)
    package.__path__ = [str(root.resolve() / "launcher")]
    sys.modules[package_name] = package
    run = importlib.import_module(package_name + ".managed_run")
    job = importlib.import_module(package_name + ".managed_job")
    return run.ManagedRun, job.ManagedJob


class LauncherContractTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ManagedJobTests.setUp
    asyncSetUp = fixtures.ManagedJobTests.asyncSetUp
    providers = fixtures.ManagedJobTests.providers
    app = fixtures.ManagedJobTests.app
    begin = fixtures.ManagedJobTests.begin
    finish = fixtures.ManagedJobTests.finish
    events = fixtures.ManagedJobTests.events
    base = fixtures.ManagedJobTests.base

    async def exercise(self, clients, *, legacy_context=False, legacy_paths=False):
        context = self.application.state.managed_context.path("w_main", self.cid)
        payload = json.loads(context.read_text())
        self.assertEqual(payload["managed_jobs_api"], "agent-v1")
        if legacy_context:
            del payload["managed_jobs_api"]
            context.write_text(json.dumps(payload))
        requests = []
        loop = asyncio.get_running_loop()

        def urlopen(request, timeout):
            self.assertEqual(timeout, 15)
            self.assertEqual(
                request.get_header("Authorization"), "Bearer " + self.token
            )
            body = json.loads(request.data)
            path = request.selector
            requests.append((path, body))
            response = asyncio.run_coroutine_threadsafe(
                self.client.post(path, json=body, headers=dict(request.header_items())),
                loop,
            ).result(timeout=5)
            if response.status_code >= 400:
                raise HTTPError(
                    request.full_url,
                    response.status_code,
                    "Agent error",
                    {},
                    io.BytesIO(response.content),
                )
            return io.BytesIO(response.content)

        run_client, job_client = clients
        kinds = (
            ("simulation", None),
            ("timing_predict", "p_predict"),
            ("kernel_profile", "kp_profile"),
            ("kernel_measure", "km_measure"),
        )
        with (
            patch.dict(
                "os.environ",
                {
                    "VIBESIM_MANAGED_JOB_CONTEXT": str(context),
                    "VIBESIM_MANAGED_RUN_CONTEXT": str(context),
                },
            ),
            patch("urllib.request.urlopen", side_effect=urlopen),
        ):
            for kind, resource in kinds:
                root = self.repo / "logs" / kind
                self.assertFalse(root.exists())
                before = len(requests)

                def invoke(kind, root, resource):
                    if kind == "simulation":
                        client = run_client.from_environment()
                        client.register(root, run_count=2, axes=["tp"])
                    else:
                        client = job_client.from_environment(kind)
                        client.register(
                            root,
                            descriptor={"source": "contract"},
                            analyzer_resource_id=resource,
                        )
                    client.report("running")
                    client.report("ready")
                    return client

                client = await asyncio.to_thread(invoke, kind, root, resource)
                prefix = (
                    (
                        "/api/internal/managed-runs"
                        if kind == "simulation"
                        else "/api/internal/managed-jobs"
                    )
                    if legacy_paths
                    else "/api/agent/v1/internal/jobs"
                )
                self.assertEqual(
                    [path for path, _ in requests[before:]],
                    [
                        prefix + "/register",
                        prefix + f"/{client.job_id}/status",
                        prefix + f"/{client.job_id}/status",
                    ],
                )
                row = self.application.state.jobs.storage("w_main").get(client.job_id)
                self.assertEqual((row["job_kind"], row["status"]), (kind, "ready"))
                self.assertEqual(
                    (row["conversation_id"], row["turn_id"]),
                    (self.cid, self.request.turn_id),
                )
                event = self.events()[-1]["payload"]
                if kind == "simulation":
                    metadata = json.loads((root / "experiment.meta.json").read_text())
                    self.assertEqual(metadata["experiment_id"], client.experiment_id)
                    self.assertEqual(event["kind"], "experiment.ready")
                    self.assertEqual(event["experimentId"], client.experiment_id)
                else:
                    self.assertEqual(event["kind"], "job.ready")
                    self.assertEqual(event["resourceId"], client.resource_id)
                    self.assertEqual(event["analyzerResourceId"], resource)

    async def test_new_clients_with_new_context(self):
        await self.exercise(self.new_clients)

    async def test_new_clients_with_legacy_context(self):
        await self.exercise(self.new_clients, legacy_context=True, legacy_paths=True)

    async def test_legacy_clients_with_new_context(self):
        await self.exercise(self.legacy_clients, legacy_paths=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher-root", type=Path, required=True)
    parser.add_argument("--legacy-launcher-root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    LauncherContractTests.new_clients = load_clients(args.launcher_root, "new_launcher")
    LauncherContractTests.legacy_clients = load_clients(
        args.legacy_launcher_root, "old_launcher"
    )
    unittest.main(argv=[sys.argv[0], *remaining])
