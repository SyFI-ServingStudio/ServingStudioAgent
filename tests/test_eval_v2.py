import asyncio
import json
import sqlite3
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

from tests import test_application as application_fixture
from tests import test_workspace_http_v2 as workspace_fixture
from vibesim_agent.domain.turns import Outcome
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.runtime.container import OWNER_LABEL


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    app = workspace_fixture.WorkspaceHttpTests.app
    git = workspace_fixture.WorkspaceHttpTests.git
    endpoint = "/api/agent/v1/tools/eval"

    def setUp(self):
        workspace_fixture.WorkspaceHttpTests.setUp(self)
        self.headers = {"Authorization": "Bearer token"}
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        self.fail_adapter = False
        self.failed_result = False
        self.observed_tokens = []

    async def asyncSetUp(self):
        await workspace_fixture.WorkspaceHttpTests.asyncSetUp(self)
        self.addAsyncCleanup(self.application.state.evaluations.close)

    def providers(self, home, environment, prompts):
        setup = application_fixture.ApplicationTests.providers(
            self, home, environment, prompts
        )
        owner = self

        class Adapter:
            adapter_id = "test"

            async def run(self, request):
                owner.calls.append(request)
                path = owner.application.state.managed_context.path(
                    request.workspace_id, request.conversation_id
                )
                token = json.loads(path.read_text())["capability_token"]
                owner.assertIsNotNone(
                    owner.application.state.capabilities.authorize(token)
                )
                owner.observed_tokens.append(token)
                owner.docker.current = {
                    "Id": "eval-container",
                    "Config": {
                        "Labels": {
                            OWNER_LABEL: json.dumps(
                                ["test", request.workspace_id, request.conversation_id]
                            )
                        }
                    },
                }
                yield {
                    "kind": "session",
                    "role": request.role.value,
                    "session_id": "saved",
                }
                yield {"kind": "role_ready", "role": request.role.value}
                owner.entered.set()
                await owner.release.wait()
                if owner.fail_adapter:
                    raise RuntimeError("private driver failure")
                if owner.failed_result:
                    yield {
                        "kind": "final",
                        "text": "",
                        "failure": {"code": "agent_call_timeout"},
                    }
                    return
                yield {
                    "kind": "final",
                    "text": '{"action":"final_answer","message":"Done"}',
                }

        registry = ProviderRegistry()
        registry.register(replace(setup.registry.provider("test"), adapter=Adapter()))
        return replace(setup, registry=registry)

    def path(self, request):
        return f"/api/agent/v1/workspaces/{request.workspace_id}/conversations/{request.conversation_id}"

    async def send(self, **body):
        return await self.client.post(
            self.endpoint, headers=self.headers, json={"prompt": "work", **body}
        )

    async def test_http_authority_defaults_and_context_cleanup_keep_workspace_and_home(
        self,
    ):
        before = self.application.state.workspaces.list()
        denied = await self.client.post(self.endpoint, json={"prompt": "work"})
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(self.application.state.workspaces.list(), before)
        response = await self.send()
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["ok"])
        self.assertTrue(result["conversation_id"].startswith("eval-"))
        self.assertEqual(result["workspace_id"], "w_" + result["conversation_id"])
        self.assertEqual(
            (result["sandbox"], result["agent_mode"], result["autonomous"]),
            ("workspace-write", "orchestrated", True),
        )
        self.assertTrue(result["kept_workspace"])
        self.assertFalse(result["kept_container"])
        request = self.calls[0]
        root = self.application.state.workspaces.conversation_runtime_path(
            request.workspace_id, request.conversation_id
        )
        self.assertTrue(root.is_dir())
        self.assertTrue(
            self.application.state.workspaces.repo_path(request.workspace_id).is_dir()
        )
        self.assertIn(["docker", "rm", "-f", "eval-container"], self.docker.calls)
        history = (await self.client.get(self.path(request))).json()
        self.assertEqual(history["messages"][-1]["content"], result["final"])
        self.assertEqual(history["messages"][-1]["outcome"], result["outcome"])
        for token in self.observed_tokens:
            self.assertIsNone(self.application.state.capabilities.authorize(token))
        self.assertFalse(
            self.application.state.managed_context.path(
                request.workspace_id, request.conversation_id
            ).exists()
        )

    async def test_keep_container_true_preserves_runtime_without_removal(self):
        response = await self.send(
            keep_container=True, agent_mode="single", autonomous=False
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["kept_container"])
        self.assertFalse(result["autonomous"])
        self.assertFalse(any(command[1] == "rm" for command in self.docker.calls))

    async def test_invalid_eval_request_does_not_create_workspace_or_start_adapter(
        self,
    ):
        registry = self.application.state.workspaces
        before = registry.list(include_archived=True)
        for body, status in (
            ({"prompt": " \n"}, 400),
            ({"agent_mode": "invalid"}, 400),
            ({"sandbox": "invalid"}, 422),
        ):
            with self.subTest(body=body):
                response = await self.send(**body)
                self.assertEqual(response.status_code, status, response.text)
                self.assertEqual(registry.list(include_archived=True), before)
                self.assertEqual(list(registry.root.glob(".workspace-*")), [])
                self.assertEqual(self.calls, [])

    async def test_cancelled_http_waiter_leaves_turn_for_targeted_browser_stop(self):
        self.release.clear()
        sending = asyncio.create_task(self.send(agent_mode="single"))
        await asyncio.wait_for(self.entered.wait(), 3)
        request = self.calls[-1]
        handle = self.application.state.turns.current(
            request.workspace_id, request.conversation_id
        )
        sending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await sending
        self.assertIs(
            self.application.state.turns.current(
                request.workspace_id, request.conversation_id
            ),
            handle,
        )
        response = await asyncio.wait_for(
            self.client.post(
                self.path(request) + "/cancel", params={"turn_id": request.turn_id}
            ),
            3,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["cancelled"])
        await asyncio.wait_for(self.application.state.evaluations.close(), 3)
        self.assertIn(["docker", "rm", "-f", "eval-container"], self.docker.calls)

    async def test_cancelled_http_during_preparation_still_runs_background_evaluation(
        self,
    ):
        evaluations = self.application.state.evaluations
        entered, release = threading.Event(), threading.Event()
        original = evaluations.workspaces.create

        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test preparation not released")
            return original(*args, **kwargs)

        with patch.object(evaluations.workspaces, "create", side_effect=blocked):
            sending = asyncio.create_task(self.send(agent_mode="single"))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                sending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await sending
                tasks = list(evaluations._tasks)
                self.assertEqual(len(tasks), 1)
            finally:
                release.set()
            result = await asyncio.wait_for(asyncio.shield(tasks[0]), 3)
        self.assertTrue(result["ok"])
        self.assertFalse(result["kept_container"])

    async def cleanup_race(
        self, *, fail_cleanup=False, fail_driver=False, failed_result=False
    ):
        evaluations = self.application.state.evaluations
        entered, release = asyncio.Event(), asyncio.Event()
        original = evaluations.remove_container
        self.fail_adapter = fail_driver
        self.failed_result = failed_result

        async def cleanup(wid, cid):
            entered.set()
            await release.wait()
            if fail_cleanup:
                raise RuntimeError("private removal failure")
            await original(wid, cid)

        with patch.object(evaluations, "remove_container", side_effect=cleanup):
            sending = asyncio.create_task(self.send(agent_mode="single"))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                request = self.calls[-1]
                self.assertTrue(
                    self.application.state.turns.cancel(
                        request.workspace_id, request.conversation_id, request.turn_id
                    )
                )
            finally:
                release.set()
            response = await asyncio.wait_for(sending, 3)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(result["ok"])
        history = (await self.client.get(self.path(request))).json()
        if fail_cleanup or fail_driver or failed_result:
            self.assertIsNone(result["outcome"])
            self.assertTrue(result["failure_code"])
            self.assertEqual(history["messages"][-1]["outcome"], "failed")
        else:
            self.assertEqual(result["outcome"], "cancelled")
            self.assertEqual(result["failure_code"], "")
            self.assertEqual(history["messages"][-1]["outcome"], "cancelled")
        self.assertNotIn("private", result["error"])
        return result

    async def test_stop_during_failed_removal_stays_failed_and_container_retained(self):
        result = await self.cleanup_race(fail_cleanup=True)
        self.assertTrue(result["kept_container"])

    async def test_driver_failure_survives_stop_during_successful_cleanup(self):
        result = await self.cleanup_race(fail_driver=True)
        self.assertFalse(result["kept_container"])

    async def test_explicit_failed_result_survives_stop_during_successful_cleanup(self):
        result = await self.cleanup_race(failed_result=True)
        self.assertFalse(result["kept_container"])
        self.assertEqual(result["failure_code"], "agent_call_timeout")

    async def test_successful_execution_stopped_during_cleanup_remains_cancelled(self):
        result = await self.cleanup_race()
        self.assertFalse(result["kept_container"])

    async def test_workspace_next_turn_cannot_run_before_eval_container_cleanup(self):
        evaluations = self.application.state.evaluations
        entered, release = asyncio.Event(), asyncio.Event()
        original = evaluations.remove_container

        async def cleanup(wid, cid):
            entered.set()
            await release.wait()
            await original(wid, cid)
            self.docker.current = None

        with patch.object(evaluations, "remove_container", side_effect=cleanup):
            sending = asyncio.create_task(self.send(agent_mode="single"))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                request = self.calls[-1]
                created = await self.client.post(
                    f"/api/agent/v1/workspaces/{request.workspace_id}/conversations",
                    json={"agentMode": "single"},
                )
                self.assertEqual(created.status_code, 200, created.text)
                handle = self.application.state.turns.start(
                    request.workspace_id, created.json()["id"], "next"
                )
                await asyncio.sleep(0)
                self.assertEqual(len(self.calls), 1)
                self.assertFalse(handle.finished)
            finally:
                release.set()
            self.assertEqual((await asyncio.wait_for(sending, 3)).status_code, 200)
            self.assertEqual(
                (
                    await asyncio.wait_for(self.application.state.turns.wait(handle), 3)
                ).outcome,
                Outcome.ANSWER,
            )

    async def test_shutdown_during_preparation_drains_thread_and_starts_no_turn(self):
        evaluations = self.application.state.evaluations
        entered, release = threading.Event(), threading.Event()
        original = evaluations.workspaces.create

        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test preparation not released")
            return original(*args, **kwargs)

        with patch.object(evaluations.workspaces, "create", side_effect=blocked):
            running = asyncio.create_task(evaluations.run("work"))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                closing = asyncio.create_task(evaluations.close())
                await asyncio.sleep(0)
                self.assertFalse(closing.done())
                closing.cancel()
                second_closing = asyncio.create_task(evaluations.close())
                await asyncio.sleep(0)
                self.assertFalse(closing.done())
                self.assertFalse(second_closing.done())
            finally:
                release.set()
            with self.assertRaisesRegex(
                RuntimeError, "closed during workspace preparation"
            ):
                await asyncio.wait_for(running, 3)
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(closing, 3)
            await asyncio.wait_for(second_closing, 3)
        self.assertEqual(self.calls, [])
        descriptors = self.application.state.workspaces.list()
        self.assertEqual(len(descriptors), 2)
        wid = next(
            item["workspace_id"]
            for item in descriptors
            if item["workspace_id"] != "w_main"
        )
        self.assertEqual(
            self.application.state.turns.storage(wid).conversations.list(), []
        )

    async def test_terminal_database_failure_propagates_after_container_cleanup(self):
        self.release.clear()
        evaluations = self.application.state.evaluations
        running = asyncio.create_task(evaluations.run("work"))
        await asyncio.wait_for(self.entered.wait(), 3)
        request = self.calls[-1]
        store = self.application.state.turns.storage(request.workspace_id)
        with patch.object(
            store.turns,
            "finish",
            side_effect=sqlite3.OperationalError("terminal write failed"),
        ):
            self.release.set()
            with self.assertRaisesRegex(
                sqlite3.OperationalError, "terminal write failed"
            ):
                await asyncio.wait_for(running, 3)
        self.assertIn(["docker", "rm", "-f", "eval-container"], self.docker.calls)
        self.assertEqual(
            store.turns.get(request.conversation_id, request.turn_id)["status"],
            "running",
        )
