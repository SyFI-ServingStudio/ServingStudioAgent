import asyncio
import unittest
from dataclasses import replace

import httpx
from pydantic import SecretStr

from tests import test_application as fixtures
from vibesim_agent.domain.turns import Outcome
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.storage.database import Database


class ToolsTests(unittest.IsolatedAsyncioTestCase):
    app = fixtures.ApplicationTests.app

    def setUp(self):
        fixtures.ApplicationTests.setUp(self)
        self.settings = self.settings.model_copy(
            update={
                "agent": self.settings.agent.model_copy(
                    update={
                        "api_token": SecretStr("test-token"),
                        "repo_root": self.root,
                    }
                )
            }
        )
        (self.root / "SKILL.md").write_text("# Agent tools\n")
        self.release, self.entered = asyncio.Event(), asyncio.Event()
        self.release.set()
        self.fail_adapter = False

    def providers(self, home, environment, prompts):
        configured = fixtures.ApplicationTests.providers(
            self, home, environment, prompts
        )
        owner = self

        class Adapter:
            adapter_id = "test"

            async def run(self, request):
                owner.calls.append(request)
                yield {
                    "kind": "session",
                    "role": request.role.value,
                    "session_id": "saved",
                }
                yield {"kind": "role_ready", "role": request.role.value}
                owner.entered.set()
                await owner.release.wait()
                if owner.fail_adapter:
                    raise RuntimeError("private backend detail")
                yield {
                    "kind": "final",
                    "text": '{"action":"request_user_input","message":"Which model?"}',
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
        self.base = "/api/agent/v1/tools/workspaces/w_main/conversations"
        self.headers = {"Authorization": "Bearer test-token"}

    async def create(self):
        response = await self.client.post(
            self.base,
            headers=self.headers,
            json={"agentMode": "single", "sandbox": "read-only", "autonomous": True},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.cid = response.json()["id"]
        return self.base + "/" + self.cid

    async def test_token_is_exact_and_skill_is_public(self):
        for authorization in (
            None,
            "Bearer wrong",
            "bearer test-token",
            "Bearer test-token ",
        ):
            headers = {} if authorization is None else {"Authorization": authorization}
            response = await self.client.post(
                self.base, headers=headers, json={"agentMode": "single"}
            )
            self.assertEqual(response.status_code, 401, response.text)
        response = await self.client.get("/api/agent/v1/tools/skill")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "# Agent tools\n")
        self.assertIn("text/markdown", response.headers["content-type"])
        self.assertFalse(self.calls)

    async def test_create_send_resume_inherits_settings_and_question_is_success(self):
        path = await self.create()
        first = await self.client.post(
            path + "/messages", headers=self.headers, json={"text": "question"}
        )
        self.assertEqual(first.status_code, 200, first.text)
        result = first.json()
        self.assertTrue(result["ok"])
        self.assertEqual(result["outcome"], "request_user_input")
        self.assertEqual(result["final"], "Which model?")
        self.assertEqual(result["sandbox"], "read-only")
        self.assertTrue(result["autonomous"])
        self.assertEqual(result["agent_mode"], "single")
        self.assertEqual(result["sessions"], {"assistant": "saved"})
        self.assertIsNone(result["failure"])
        self.assertEqual(result["citations"], [])
        second = await self.client.post(
            path + "/messages",
            headers=self.headers,
            json={
                "text": "another",
                "autonomous_mode": False,
                "agentMode": "orchestrated",
            },
        )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertTrue(second.json()["autonomous"])
        self.assertEqual(second.json()["agent_mode"], "single")
        self.assertEqual(
            [request.session_id for request in self.calls], [None, "saved"]
        )
        history = await self.client.get(path, headers=self.headers)
        self.assertEqual(len(history.json()["messages"]), 4)
        browser = await self.client.get(path.replace("/tools", ""))
        self.assertEqual(browser.json()["messages"], history.json()["messages"])

    async def test_cancelled_http_waiter_keeps_turn_then_browser_can_stop_it(self):
        path = await self.create()
        self.release.clear()
        sending = asyncio.create_task(
            self.client.post(
                path + "/messages", headers=self.headers, json={"text": "work"}
            )
        )
        try:
            await asyncio.wait_for(self.entered.wait(), 3)
            handle = self.application.state.turns.current("w_main", self.cid)
            sending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await sending
            self.assertIs(
                self.application.state.turns.current("w_main", self.cid), handle
            )
            self.assertFalse(handle.task.done())
            duplicate = await self.client.post(
                path + "/messages", headers=self.headers, json={"text": "duplicate"}
            )
            self.assertEqual(duplicate.status_code, 409)
            stopped = await self.client.post(
                path.replace("/tools", "") + "/cancel",
                params={"turn_id": handle.request.turn_id},
            )
            self.assertTrue(stopped.json()["cancelled"])
            self.assertEqual(
                (await self.application.state.turns.wait(handle)).outcome,
                Outcome.CANCELLED,
            )
            self.assertFalse(
                self.application.state.managed_context.path("w_main", self.cid).exists()
            )
        finally:
            self.release.set()
            if not sending.done():
                sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)

    async def test_runtime_failure_is_not_success_despite_nonempty_final(self):
        path = await self.create()
        self.fail_adapter = True
        response = await self.client.post(
            path + "/messages", headers=self.headers, json={"text": "fail"}
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertFalse(result["ok"])
        self.assertTrue(result["final"])
        self.assertIsNone(result["outcome"])
        self.assertEqual(result["failure_code"], "agent_runtime_failure")
        self.assertEqual(result["error"], result["failure"]["message"])
        self.assertNotIn("private backend detail", response.text)

    async def test_invalid_requests_do_not_create_turns(self):
        path = await self.create()
        for body, status in (
            ({"text": " "}, 400),
            ({"text": "x", "agentMode": "unknown"}, 400),
            ({"text": "x", "sandbox_mode": "unknown"}, 422),
        ):
            response = await self.client.post(
                path + "/messages", headers=self.headers, json=body
            )
            self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(
            self.application.state.turns.storage("w_main").turns.list(self.cid), []
        )

    async def test_empty_configured_token_opens_tools(self):
        self.settings = self.settings.model_copy(
            update={
                "agent": self.settings.agent.model_copy(
                    update={"api_token": SecretStr("")}
                )
            }
        )
        app = self.app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(self.base, json={"agentMode": "single"})
        self.assertEqual(response.status_code, 200, response.text)
