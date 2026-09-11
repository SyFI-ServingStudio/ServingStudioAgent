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


class RuntimeProviderWireTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ApplicationTests.setUp
    app = fixtures.ApplicationTests.app
    base = "/api/agent/v1/workspaces/w_main/conversations"

    def providers(self, home, environment, prompts):
        setup = fixtures.ApplicationTests.providers(self, home, environment, prompts)
        original = setup.registry.provider("test")
        registry = ProviderRegistry()
        for provider_id in ("test", "connection_b", "unavailable"):
            registry.register(
                replace(
                    original,
                    provider_id=provider_id,
                    session_scope=provider_id + ":scope",
                    settings=ProviderSettings(
                        model="test-model",
                        effort="high" if provider_id == "test" else "low",
                        service_tier="default" if provider_id == "test" else "fast",
                    ),
                    catalog=lambda: (
                        Model(
                            "test-model",
                            "Shared",
                            ("low", "high"),
                            "high",
                            ("default", "fast"),
                        ),
                    ),
                    credentials=Credentials(all_secrets=("MISSING",))
                    if provider_id == "unavailable"
                    else Credentials(),
                )
            )
        registry.register(
            replace(
                original,
                provider_id="unique",
                session_scope="unique:scope",
                settings=ProviderSettings(model="unique-model", effort="high"),
                catalog=lambda: (Model("unique-model", "Unique", ("high",), "high"),),
            )
        )
        return replace(setup, registry=registry)

    async def asyncSetUp(self):
        Database.create(self.state / "workspace.sqlite")
        self.application = self.app()
        self.store = self.application.state.turns.storage("w_main")
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.application), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)

    async def create(self, override=None):
        response = await self.client.post(
            self.base,
            json={
                "agentMode": "single",
                "codex_runtime": {"assistant": override or {}},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_shared_models_use_explicit_connection_and_its_defaults(self):
        created = await self.create({"provider": "connection_b"})
        expected = {
            "provider": "connection_b",
            "model": "test-model",
            "effort": "low",
            "serviceTier": "fast",
        }
        self.assertEqual(created["codex_runtime"]["assistant"], expected)
        stored = self.store.conversations.runtimes(created["id"])[Role.ASSISTANT]
        self.assertEqual(
            (stored.provider_id, stored.model_id, stored.session_scope),
            ("connection_b", "test-model", "connection_b:scope"),
        )
        history = await self.client.get(self.base + "/" + created["id"])
        self.assertEqual(history.json()["codex_runtime"]["assistant"], expected)
        catalog = (await self.client.get("/api/agent/v1/codex-backends")).json()
        self.assertEqual(catalog["defaults"]["assistant"]["provider"], "test")
        self.assertEqual(
            {
                model["family"]
                for model in catalog["models"]
                if model["id"] == "test-model"
            },
            {"test", "connection_b", "unavailable"},
        )

    async def test_legacy_selection_and_unique_model_payload_stay_compatible(self):
        created = await self.create({"model": "legacy-model"})
        self.assertEqual(created["codex_runtime"]["assistant"]["provider"], "test")
        unique = await self.create({"model": "unique-model"})
        self.assertEqual(
            unique["codex_runtime"]["assistant"],
            {
                "model": "unique-model",
                "effort": "high",
                "serviceTier": "default",
            },
        )

    async def test_provider_model_mismatch_unknown_and_unavailable_never_create(self):
        before = self.store.conversations.list()
        for override, status in (
            ({"provider": "connection_b", "model": "unique-model"}, 400),
            ({"provider": "unknown"}, 400),
            ({"provider": "unavailable"}, 409),
            ({"provider": ""}, 422),
        ):
            with self.subTest(override=override):
                response = await self.client.post(
                    self.base,
                    json={
                        "agentMode": "single",
                        "codex_runtime": {"assistant": override},
                    },
                )
                self.assertEqual(response.status_code, status, response.text)
                self.assertEqual(self.store.conversations.list(), before)

    async def test_started_connection_cannot_switch_even_with_identical_model(self):
        created = await self.create({"provider": "connection_b"})
        cid = created["id"]
        self.store.conversations.append(cid, "user", "existing history")
        self.store.sessions.save(
            cid,
            Session(Role.ASSISTANT, "connection_b", "connection_b:scope", "original"),
        )
        before = (
            self.store.conversations.get(cid),
            self.store.conversations.runtimes(cid),
            self.store.sessions.list(cid),
        )
        response = await self.client.patch(
            self.base + "/" + cid + "/runtime",
            json={
                "codex_runtime": {
                    "assistant": {"provider": "test", "model": "test-model"}
                },
            },
        )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(
            response.json()["detail"]["code"], "conversation_runtime_locked"
        )
        self.assertEqual(
            (
                self.store.conversations.get(cid),
                self.store.conversations.runtimes(cid),
                self.store.sessions.list(cid),
            ),
            before,
        )
        response = await self.client.patch(
            self.base + "/" + cid + "/runtime",
            json={
                "codex_runtime": {
                    "assistant": {
                        "provider": "connection_b",
                        "model": "test-model",
                        "effort": "high",
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.store.sessions.list(cid), before[2])
        self.assertEqual(
            response.json()["codex_runtime"]["assistant"]["provider"], "connection_b"
        )
        self.assertEqual(
            response.json()["codex_runtime"]["orchestrator"]["provider"], "test"
        )
