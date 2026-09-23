import asyncio
import json
import sqlite3
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx

from tests import test_application as application_fixture
from vibesim_agent.application import build_application
from vibesim_agent.domain.roles import Role
from vibesim_agent.domain.turns import Outcome
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.storage.database import Database


class ManagedLifecycleTests(unittest.IsolatedAsyncioTestCase):
    setUp = application_fixture.ApplicationTests.setUp
    base = "/api/agent/v1/workspaces/w_main/conversations"

    def providers(self, home, environment, prompts):
        configured = application_fixture.ApplicationTests.providers(
            self, home, environment, prompts
        )
        owner = self

        class Adapter:
            adapter_id = "test"

            async def run(self, request):
                owner.observe(request)
                owner.calls.append((request, home(request)))
                try:
                    yield {
                        "kind": "session",
                        "role": request.role.value,
                        "session_id": "saved-" + request.role.value,
                    }
                    yield {"kind": "role_ready", "role": request.role.value}
                    owner.entered.set()
                    if owner.behavior == "block":
                        await asyncio.Event().wait()
                    if owner.behavior == "fail":
                        raise RuntimeError("adapter failed")
                    response = {"action": "final_answer", "message": "Done"}
                    if owner.behavior == "delegate" and len(owner.calls) == 1:
                        response = {"action": "delegate", "task": "inspect"}
                    yield {"kind": "final", "text": json.dumps(response)}
                finally:
                    owner.closed_authority.append(
                        owner.app.state.capabilities.authorize(owner.tokens[-1])
                        is not None
                    )

        registry = ProviderRegistry()
        registry.register(
            replace(configured.registry.provider("test"), adapter=Adapter())
        )
        return replace(configured, registry=registry)

    def observe(self, request):
        path = self.context_path(request.conversation_id)
        data = json.loads(path.read_text())
        token = data["capability_token"]
        capability = self.app.state.capabilities.authorize(token)
        self.assertIsNotNone(capability)
        self.assertEqual(
            (
                capability.workspace_id,
                capability.conversation_id,
                capability.turn_id,
                capability.role,
            ),
            (
                request.workspace_id,
                request.conversation_id,
                request.turn_id,
                request.role.value,
            ),
        )
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        if token not in self.tokens:
            self.tokens.append(token)
            self.inodes.append(path.stat().st_ino)

    def context_path(self, conversation_id):
        return self.state / "runtime" / conversation_id / "managed/context.json"

    async def setup_app(self, *, behavior="answer", before_failure=False):
        Database.create(self.state / "workspace.sqlite")
        self.behavior = behavior
        self.tokens = []
        self.inodes = []
        self.closed_authority = []
        self.entered = asyncio.Event()

        async def before(request):
            self.observe(request)
            if before_failure:
                raise RuntimeError("before call failed")

        self.app = build_application(
            self.settings,
            providers=self.providers,
            before_call=before,
            prompts_directory=self.root / "prompts",
            mcp_directory=self.mcp,
            managed_context="/runtime/context.json",
            namespace="test",
            submodules=(),
            run=self.docker,
        )
        self.addAsyncCleanup(self.app.state.turns.close)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)

    async def create(self, mode="single"):
        response = await self.client.post(self.base, json={"agentMode": mode})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["id"]

    async def send(self, conversation_id):
        response = await self.client.post(
            self.base + "/" + conversation_id + "/messages", json={"text": "question"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("event: done", response.text)
        return (await self.client.get(self.base + "/" + conversation_id)).json()

    def assert_revoked(self, conversation_id, *, removed=True):
        self.assertTrue(self.tokens)
        for token in self.tokens:
            self.assertIsNone(self.app.state.capabilities.authorize(token))
        self.assertEqual(self.context_path(conversation_id).exists(), not removed)

    async def test_success_resume_and_context_path_delivered_to_the_turn(self):
        await self.setup_app()
        cid = await self.create()
        for _ in range(2):
            history = await self.send(cid)
            self.assertEqual(history["messages"][-1]["outcome"], Outcome.ANSWER.value)
            self.assert_revoked(cid)
        self.assertEqual(len(self.tokens), 2)
        self.assertEqual(
            [request.session_id for request, _ in self.calls], [None, "saved-assistant"]
        )
        # `w_main` is external, so these turns run here and the capability file
        # is reached by path rather than through a read-only bind mount. The
        # container form of the same delivery is in test_runtime_service.
        self.assertEqual(self.docker.calls, [])
        for request, _ in self.calls:
            self.assertEqual(request.execution.managed_context, str(self.context_path(cid)))
        self.assertEqual(self.closed_authority, [True, True])

    async def test_two_roles_replace_visible_context_then_revoke_all(self):
        await self.setup_app(behavior="delegate")
        cid = await self.create("orchestrated")
        history = await self.send(cid)
        self.assertEqual(history["messages"][-1]["outcome"], Outcome.ANSWER.value)
        self.assertEqual(
            [r.role for r, _ in self.calls],
            [Role.ORCHESTRATOR, Role.IMPLEMENTER, Role.ORCHESTRATOR],
        )
        self.assertEqual(len(self.tokens), 3)
        self.assertNotEqual(self.inodes[0], self.inodes[1])
        self.assert_revoked(cid)

    async def test_before_call_failure_revokes_without_adapter(self):
        await self.setup_app(before_failure=True)
        cid = await self.create()
        history = await self.send(cid)
        self.assertEqual(history["messages"][-1]["outcome"], "failed")
        self.assertEqual(self.calls, [])
        self.assert_revoked(cid)

    async def test_adapter_failure_closes_before_revoke(self):
        await self.setup_app(behavior="fail")
        cid = await self.create()
        history = await self.send(cid)
        self.assertEqual(history["messages"][-1]["outcome"], "failed")
        self.assertEqual(self.closed_authority, [True])
        self.assert_revoked(cid)

    async def test_cancellation_closes_adapter_and_revokes(self):
        await self.setup_app(behavior="block")
        cid = await self.create()
        turns = self.app.state.turns
        handle = turns.start("w_main", cid, "question")
        await asyncio.wait_for(self.entered.wait(), 3)
        self.assertTrue(turns.cancel("w_main", cid, handle.request.turn_id))
        result = await asyncio.wait_for(turns.wait(handle), 3)
        self.assertEqual(result.outcome, Outcome.CANCELLED)
        self.assertEqual(self.closed_authority, [True])
        self.assert_revoked(cid)

    async def test_context_remove_failure_still_revokes_and_fails_turn(self):
        await self.setup_app()
        cid = await self.create()
        with patch.object(
            self.app.state.managed_context,
            "remove",
            side_effect=OSError("remove failed"),
        ):
            history = await self.send(cid)
        self.assertEqual(history["messages"][-1]["outcome"], "failed")
        self.assert_revoked(cid, removed=False)

    async def test_storage_finish_failure_cannot_leave_live_capability(self):
        await self.setup_app()
        cid = await self.create()
        turns = self.app.state.turns
        with patch.object(
            turns.storage("w_main").turns,
            "finish",
            side_effect=sqlite3.OperationalError("finish failed"),
        ):
            handle = turns.start("w_main", cid, "question")
            with self.assertRaisesRegex(sqlite3.OperationalError, "finish failed"):
                await turns.wait(handle)
        self.assert_revoked(cid)
        self.assertEqual(
            turns.storage("w_main").turns.get(cid, handle.request.turn_id)["status"],
            "running",
        )

    async def test_managed_directory_symlink_fails_before_profiles_or_docker(self):
        await self.setup_app()
        cid = await self.create()
        parent = self.context_path(cid).parent.parent
        parent.mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        (parent / "managed").symlink_to(outside, target_is_directory=True)
        history = await self.send(cid)
        self.assertEqual(history["messages"][-1]["outcome"], "failed")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual([p.name for p in parent.iterdir()], ["managed"])
        self.assertEqual(self.docker.calls, [])
        self.assertEqual(self.tokens, [])

    def test_overlapping_target_rejected_before_any_application_writes(self):
        for target in (
            "/workspace/context.json",
            "//workspace/context.json",
            "/model/context.json",
            "/opt/context.json",
            "/opt/vibesim/prompts/nested/context.json",
            "relative/context.json",
            "/context.json",
            "/runtime/../context.json",
            "/runtime/context.json/",
            str(self.settings.container.home / "context.json"),
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                build_application(
                    self.settings,
                    providers=self.providers,
                    prompts_directory=self.root / "prompts",
                    mcp_directory=self.mcp,
                    managed_context=target,
                    namespace="test",
                    submodules=(),
                    run=self.docker,
                )
        self.assertFalse((self.root / "prompts").exists())
        self.assertEqual(self.factories, 0)
        self.assertEqual(self.docker.calls, [])
