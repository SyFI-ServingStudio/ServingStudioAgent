"""Browser conversation collection wire format and explicit composition."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock

import httpx

from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import Role
from vibesim_agent.main import create_app
from vibesim_agent.providers.base import Credentials, Model, Provider
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.services.conversation import ConversationService
from vibesim_agent.services.turn import TurnStorage
from vibesim_agent.settings import ProviderSettings
from vibesim_agent.storage.conversations import Conversations
from vibesim_agent.storage.database import Database
from vibesim_agent.storage.sessions import Sessions
from vibesim_agent.storage.turns import Turns


class ConversationHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        database = Database.create(self.root / "state.sqlite")
        self.store = TurnStorage(
            Conversations(database), Sessions(database), Turns(database)
        )

        def storage(workspace_id):
            if workspace_id == "invalid":
                raise ValueError("invalid workspace ID")
            if workspace_id != "w":
                raise KeyError(workspace_id)
            return self.store

        self.adapter = Mock(adapter_id="test-cli")
        self.providers = ProviderRegistry()
        self.register("first", "model")
        self.register("second", "other")
        self.register("missing", "unavailable", Credentials(all_secrets=("TOKEN",)))
        default = RoleRuntime("first", "first:v1", "model", "low", "default")
        self.runtimes = {role: default for role in Role}
        self.fingerprint = Mock(return_value="explicit-fingerprint")
        self.service = ConversationService(
            storage,
            providers=self.providers,
            default_runtimes=self.runtimes,
            fingerprint=self.fingerprint,
        )
        self.turns = Mock(storage=storage, close=AsyncMock())
        self.client = self.make_client(conversations=self.service)
        self.path = "/api/agent/v1/workspaces/w/conversations"

    def register(self, name, model_id, credentials=None):
        self.providers.register(
            Provider(
                name,
                name,
                self.adapter,
                ProviderSettings(model=model_id, effort="high"),
                name + ":v1",
                lambda: (
                    Model(
                        model_id, model_id, ("low", "high"), "high", ("default", "fast")
                    ),
                ),
                credentials if credentials is not None else Credentials(),
            )
        )

    def make_client(self, **kwargs):
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.turns, **kwargs)),
            base_url="http://test",
        )
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_defaults_create_list_and_history_without_runtime_launch(self):
        response = await self.client.post(self.path, json={})
        self.assertEqual(response.status_code, 200, response.text)
        created = response.json()
        self.assertEqual(created["agent_mode"], "orchestrated")
        self.assertEqual(created["sandbox"], "workspace-write")
        self.assertIs(created["autonomous"], False)
        self.assertEqual(created["prompt_fingerprint"], "explicit-fingerprint")
        self.assertEqual(
            created["codex_runtime"],
            {
                role.value: {
                    "model": "model",
                    "effort": "low",
                    "serviceTier": "default",
                }
                for role in Role
            },
        )
        self.assertEqual(
            (await self.client.get(self.path + "/" + created["id"])).json(), created
        )
        self.assertEqual(
            (await self.client.get(self.path)).json(),
            {
                "workspace_id": "w",
                "sandbox_modes": ["read-only", "workspace-write", "danger-full-access"],
                "conversations": [
                    {
                        key: created[key]
                        for key in ("id", "title", "naming_state", "updated_at")
                    }
                ],
            },
        )
        self.adapter.run.assert_not_called()
        self.turns.start.assert_not_called()
        self.fingerprint.assert_called_once()

    async def test_catalog_projects_registered_providers_without_starting_runtime(self):
        self.register("extra", "extra-model", Credentials(any_secrets=("ONE", "TWO")))
        response = await self.client.get("/api/agent/v1/codex-backends")
        self.assertEqual(response.status_code, 200, response.text)
        catalog = response.json()
        self.assertEqual(
            catalog["defaults"],
            {
                role.value: {
                    "model": "model",
                    "effort": "low",
                    "serviceTier": "default",
                }
                for role in Role
            },
        )
        self.assertEqual(
            catalog["families"][-1],
            {
                "id": "extra",
                "label": "extra",
                "runner": "test-cli",
                "available": False,
                "requiredEnvironment": [],
                "credentialEnvironmentAlternatives": ["ONE", "TWO"],
            },
        )
        self.assertEqual(
            catalog["models"][-1],
            {
                "id": "extra-model",
                "label": "extra-model",
                "family": "extra",
                "familyLabel": "extra",
                "runner": "test-cli",
                "available": False,
                "efforts": ["low", "high"],
                "defaultEffort": "high",
                "serviceTiers": ["default", "fast"],
                "defaultServiceTier": "default",
            },
        )
        self.adapter.run.assert_not_called()
        self.turns.start.assert_not_called()
        self.assertEqual(self.store.conversations.list(), [])

    async def test_unconfigured_catalog_is_explicitly_unavailable(self):
        client = self.make_client()
        response = await client.get("/api/agent/v1/codex-backends")
        self.assertEqual(response.status_code, 503)

    async def test_aliases_partial_overrides_and_legacy_normalization(self):
        for alias in ("agentMode", "agent_mode"):
            response = await self.client.post(
                self.path,
                json={
                    alias: "single",
                    "sandbox": "read-only",
                    "autonomous": True,
                    "codex_runtime": {
                        "assistant": {
                            "model": "other",
                            "effort": "max",
                            "serviceTier": "unsupported",
                        },
                        "implementer": {"service_tier": "fast"},
                    },
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["agent_mode"], "single")
            self.assertEqual(
                result["codex_runtime"]["assistant"],
                {"model": "other", "effort": "high", "serviceTier": "default"},
            )
            self.assertEqual(
                result["codex_runtime"]["implementer"]["serviceTier"], "fast"
            )
            self.assertEqual(
                self.store.conversations.runtimes(result["id"])[
                    Role.ASSISTANT
                ].provider_id,
                "second",
            )

    async def test_unknown_model_mode_and_credentials_have_legacy_errors(self):
        cases = [
            (
                {"codex_runtime": {"assistant": {"model": "unknown"}}},
                400,
                {"code": "unknown_codex_model", "model": "unknown"},
            ),
            (
                {"agentMode": "bad"},
                400,
                {"code": "unknown_agent_mode", "agent_mode": "bad"},
            ),
            (
                {
                    "agentMode": "single",
                    "codex_runtime": {"assistant": {"model": "unavailable"}},
                },
                409,
                {"code": "codex_family_unavailable", "families": ["missing"]},
            ),
        ]
        for body, status, detail in cases:
            response = await self.client.post(self.path, json=body)
            self.assertEqual(response.status_code, status)
            self.assertEqual(response.json()["detail"], detail)
        self.assertEqual(self.service.list("w"), [])
        success = await self.client.post(
            self.path, json={"codex_runtime": {"assistant": {"model": "unavailable"}}}
        )
        self.assertEqual(success.status_code, 200)

    async def test_invalid_workspace_and_body_do_not_create(self):
        for workspace in ("missing", "invalid"):
            path = self.path.replace("/w/", f"/{workspace}/")
            for response in (
                await self.client.get(path),
                await self.client.post(path, json={}),
            ):
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json(), {"detail": "workspace not found"})
        for body in (
            {"codex_runtime": None},
            {"codex_runtime": {"assistant": {"model": None}}},
            {"agentMode": None},
            {"sandbox": "unsupported"},
        ):
            self.assertEqual(
                (await self.client.post(self.path, json=body)).status_code, 422
            )
        self.assertEqual(self.service.list("w"), [])

    async def test_peer_path_is_persisted_without_materialization_or_existence_check(
        self,
    ):
        peer = str(self.root / "does-not-exist")
        response = await self.client.post(self.path, json={"peer_workspace": peer})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["peer_workspace"], peer)
        self.assertFalse(Path(peer).exists())

    async def test_eager_requires_explicit_dependency_and_returns_its_path(self):
        rejected = await self.client.post(self.path, json={"eager": True})
        self.assertEqual(rejected.status_code, 409)
        self.assertEqual(self.service.list("w"), [])
        prepare = Mock(return_value=self.root / "repo")
        client = self.make_client(conversations=self.service, prepare_workspace=prepare)
        lazy = await client.post(self.path, json={})
        self.assertNotIn("workspace_path", lazy.json())
        prepare.assert_not_called()
        eager = await client.post(self.path, json={"eager": True})
        self.assertEqual(eager.status_code, 200)
        self.assertEqual(eager.json()["workspace_path"], str(self.root / "repo"))
        prepare.assert_called_once_with("w")

    async def test_existing_composition_lists_but_creation_requires_configuration(self):
        client = self.make_client()
        self.assertEqual((await client.get(self.path)).status_code, 200)
        response = await client.post(self.path, json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("configured providers", response.json()["detail"])

    async def test_shared_model_ids_keep_default_provider_and_reject_ambiguity(self):
        self.register("same", "model")
        created = (await self.client.post(self.path, json={})).json()
        self.assertEqual(
            self.store.conversations.runtimes(created["id"])[
                Role.ASSISTANT
            ].provider_id,
            "first",
        )
        self.register("also_second", "other")
        response = await self.client.post(
            self.path, json={"codex_runtime": {"assistant": {"model": "other"}}}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("ambiguous provider", response.json()["detail"])
        self.assertEqual(len(self.service.list("w")), 1)

    async def test_target_model_normalization_does_not_validate_provider_default_tier(
        self,
    ):
        self.providers.register(
            Provider(
                "mixed",
                "Mixed",
                self.adapter,
                ProviderSettings(
                    model="fast-model", effort="high", service_tier="fast"
                ),
                "mixed:v1",
                lambda: (
                    Model("fast-model", "Fast", ("high",), "high", ("default", "fast")),
                    Model("plain-model", "Plain", ("low",), "low"),
                ),
            )
        )
        for override in (
            {"model": "plain-model"},
            {"model": "plain-model", "serviceTier": "default"},
            {"model": "plain-model", "effort": "high", "serviceTier": "fast"},
        ):
            response = await self.client.post(
                self.path,
                json={
                    "codex_runtime": {"assistant": override},
                    "agentMode": "single",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(
                response.json()["codex_runtime"]["assistant"],
                {
                    "model": "plain-model",
                    "effort": "low",
                    "serviceTier": "default",
                },
            )
