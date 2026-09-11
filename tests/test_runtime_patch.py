import asyncio
import unittest
from dataclasses import replace

import httpx

from tests import test_application as fixtures
from vibesim_agent.domain.roles import Role
from vibesim_agent.providers.base import Credentials, Model
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.settings import ProviderSettings
from vibesim_agent.storage.database import Database
from vibesim_agent.storage.sessions import Session


class RuntimePatchTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ApplicationTests.setUp
    app = fixtures.ApplicationTests.app
    base = "/api/agent/v1/workspaces/w_main/conversations"

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
                yield {
                    "kind": "final",
                    "text": '{"action":"final_answer","message":"Done"}',
                }

        registry = ProviderRegistry()
        primary = replace(
            configured.registry.provider("test"),
            adapter=Adapter(),
            catalog=lambda: (
                Model("test-model", "Default", ("high",), "high"),
                Model(
                    "sibling-model",
                    "Sibling",
                    ("low", "high"),
                    "low",
                    service_tiers=("default", "fast"),
                ),
            ),
        )
        registry.register(primary)
        for provider_id, model_id, credentials in (
            ("other", "other-model", Credentials()),
            ("missing", "missing-model", Credentials(all_secrets=("MISSING",))),
        ):
            registry.register(
                replace(
                    primary,
                    provider_id=provider_id,
                    settings=ProviderSettings(model=model_id, effort="high"),
                    session_scope=provider_id + ":scope",
                    catalog=lambda model_id=model_id: (
                        Model(model_id, model_id, ("high",), "high"),
                    ),
                    credentials=credentials,
                )
            )
        return replace(configured, registry=registry)

    async def asyncSetUp(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        Database.create(self.state / "workspace.sqlite")
        self.application = self.app()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.application), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        created = await self.client.post(self.base, json={"agentMode": "single"})
        self.assertEqual(created.status_code, 200, created.text)
        self.cid = created.json()["id"]
        self.path = self.base + "/" + self.cid
        self.store = self.application.state.turns.storage("w_main")

    async def update(self, overrides):
        return await self.client.patch(
            self.path + "/runtime", json={"codex_runtime": overrides}
        )

    def save_sessions(self):
        for role in Role:
            self.store.sessions.save(
                self.cid, Session(role, "test", "test:scope", "saved-" + role.value)
            )

    def snapshot(self):
        return (
            self.store.conversations.get(self.cid),
            self.store.conversations.runtimes(self.cid),
            self.store.sessions.list(self.cid),
        )

    async def test_required_dto_unknown_model_missing_identity_and_inactive_credentials(
        self,
    ):
        before = self.snapshot()
        missing = await self.client.patch(self.path + "/runtime", json={})
        self.assertEqual(missing.status_code, 422)
        unknown = await self.update({"assistant": {"model": "unknown"}})
        self.assertEqual(unknown.status_code, 400)
        self.assertEqual(
            unknown.json()["detail"],
            {"code": "unknown_codex_model", "model": "unknown"},
        )
        for role in ("assistant", "implementer"):
            response = await self.update({role: {"model": "missing-model"}})
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json()["detail"]["families"], ["missing"])
        for path in (
            self.base + "/missing/runtime",
            self.path.replace("w_main", "w_missing") + "/runtime",
        ):
            response = await self.client.patch(path, json={"codex_runtime": {}})
            self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(self.snapshot(), before)

    async def test_patch_replaces_all_roles_and_rejects_invalid_options(
        self,
    ):
        first = await self.update(
            {
                "orchestrator": {
                    "model": "sibling-model",
                    "effort": "low",
                    "serviceTier": "fast",
                }
            }
        )
        self.assertEqual(first.status_code, 200, first.text)
        second = await self.update(
            {
                "assistant": {
                    "model": "sibling-model",
                    "effort": "unsupported",
                    "serviceTier": "unsupported",
                }
            }
        )
        self.assertEqual(second.status_code, 400, second.text)
        runtimes = self.store.conversations.runtimes(self.cid)
        self.assertEqual(
            (
                runtimes[Role.ASSISTANT].model_id,
                runtimes[Role.ASSISTANT].effort,
                runtimes[Role.ASSISTANT].service_tier,
            ),
            ("test-model", "high", "default"),
        )
        self.assertEqual(
            (
                runtimes[Role.ORCHESTRATOR].model_id,
                runtimes[Role.ORCHESTRATOR].effort,
                runtimes[Role.ORCHESTRATOR].service_tier,
            ),
            ("sibling-model", "low", "fast"),
        )
        self.assertEqual(
            (
                runtimes[Role.IMPLEMENTER].model_id,
                runtimes[Role.IMPLEMENTER].effort,
                runtimes[Role.IMPLEMENTER].service_tier,
            ),
            ("test-model", "high", "default"),
        )

    async def test_same_scope_runtime_change_preserves_sessions_and_touches_timestamp(
        self,
    ):
        self.save_sessions()
        self.store.conversations.append(self.cid, "user", "existing history")
        before = self.snapshot()
        response = await self.update(
            {
                "assistant": {
                    "model": "sibling-model",
                    "effort": "low",
                    "serviceTier": "fast",
                }
            }
        )
        self.assertEqual(response.status_code, 200, response.text)
        after = self.snapshot()
        self.assertEqual(after[2], before[2])
        self.assertGreater(after[0]["updated_at"], before[0]["updated_at"])
        self.assertEqual(after[1][Role.ASSISTANT].session_scope, "test:scope")
        self.assertEqual(after[1][Role.ASSISTANT].service_tier, "fast")

    async def test_inactive_scope_change_clears_only_its_session(self):
        self.save_sessions()
        self.store.conversations.append(self.cid, "user", "existing history")
        response = await self.update({"implementer": {"model": "other-model"}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            {session.role for session in self.store.sessions.list(self.cid)},
            {Role.ASSISTANT, Role.ORCHESTRATOR},
        )
        self.assertEqual(
            self.store.conversations.runtimes(self.cid)[Role.IMPLEMENTER].session_scope,
            "other:scope",
        )

    async def test_active_scope_conflict_rolls_back_earlier_inactive_updates_and_session_deletion(
        self,
    ):
        self.save_sessions()
        self.store.conversations.append(self.cid, "user", "existing history")
        before = self.snapshot()
        response = await self.update(
            {
                "orchestrator": {"model": "other-model"},
                "assistant": {"model": "other-model"},
            }
        )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(
            response.json()["detail"]["code"], "conversation_runtime_locked"
        )
        self.assertEqual(self.snapshot(), before)

    async def test_running_request_uses_snapshot_then_next_turn_uses_new_runtime_and_resume(
        self,
    ):
        self.release.clear()
        turns = self.application.state.turns
        first = turns.start("w_main", self.cid, "first")
        await asyncio.wait_for(self.entered.wait(), 3)
        request = self.calls[0]
        try:
            response = await self.update(
                {
                    "assistant": {
                        "model": "sibling-model",
                        "effort": "low",
                        "serviceTier": "fast",
                    }
                }
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(
                (
                    request.selection.model.model_id,
                    request.selection.effort,
                    request.selection.service_tier,
                ),
                ("test-model", "high", "default"),
            )
            self.assertEqual(
                first.request.runtimes[Role.ASSISTANT].model_id, "test-model"
            )
        finally:
            self.release.set()
        await asyncio.wait_for(turns.wait(first), 3)
        second = turns.start("w_main", self.cid, "second")
        await asyncio.wait_for(turns.wait(second), 3)
        next_request = self.calls[1]
        self.assertEqual(
            (
                next_request.selection.model.model_id,
                next_request.selection.effort,
                next_request.selection.service_tier,
                next_request.session_id,
            ),
            ("sibling-model", "low", "fast", "saved"),
        )
