"""Conversation creation validates selections before persisting browser state."""

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.errors import ProviderUnavailable
from vibesim_agent.domain.roles import AgentMode, Role, Sandbox
from vibesim_agent.providers.base import Credentials, Model, Provider
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.services.conversation import ConversationService
from vibesim_agent.services.turn import TurnStorage
from vibesim_agent.settings import ProviderSettings
from vibesim_agent.storage.conversations import Conversations
from vibesim_agent.storage.database import Database
from vibesim_agent.storage.sessions import Sessions
from vibesim_agent.storage.turns import Turns


class ConversationServiceTests(unittest.TestCase):
    def setUp(self):
        directory = Path(self.enterContext(TemporaryDirectory()))
        self.database = Database.create(directory / "state.sqlite")
        self.store = TurnStorage(
            Conversations(self.database), Sessions(self.database), Turns(self.database)
        )
        self.adapter = Mock()
        self.providers = ProviderRegistry()
        model = Model("model", "Model", ("low", "high"), "high")
        for name, credentials in (
            ("available", Credentials()),
            ("missing", Credentials(all_secrets=("TOKEN",))),
        ):
            self.providers.register(
                Provider(
                    name,
                    name,
                    self.adapter,
                    ProviderSettings(model="model", effort="high"),
                    name + ":v1",
                    lambda: (model,),
                    credentials,
                )
            )
        self.runtime = RoleRuntime(
            "available", "available:v1", "model", "high", "default"
        )
        self.runtimes = {role: self.runtime for role in Role}
        self.service = ConversationService(
            lambda _: self.store,
            providers=self.providers,
            default_runtimes=self.runtimes,
        )

    def create(self, **kwargs):
        options = {
            "mode": AgentMode.SINGLE,
            "sandbox": Sandbox.READ_ONLY,
            "autonomous": True,
            "peer_workspace": "peer",
        }
        options.update(kwargs)
        return self.service.create("workspace", **options)

    def test_creation_preserves_browser_shape_and_all_role_settings(self):
        result = self.create(prompt_fingerprint="fingerprint")
        self.assertRegex(result["id"], r"^[0-9a-f]{12}$")
        self.assertEqual(result["title"], "New chat")
        self.assertEqual(result["naming_state"], "pending")
        self.assertEqual(result["agent_mode"], "single")
        self.assertEqual(result["sandbox"], "read-only")
        self.assertIs(result["autonomous"], True)
        self.assertEqual(result["peer_workspace"], "peer")
        self.assertEqual(result["prompt_fingerprint"], "fingerprint")
        self.assertEqual(result["messages"], [])
        self.assertEqual(result["codex_sessions"], {})
        self.assertEqual(
            result["codex_runtime"],
            {
                role.value: {
                    "provider": "available",
                    "model": "model",
                    "effort": "high",
                    "serviceTier": "default",
                }
                for role in Role
            },
        )
        self.assertEqual(self.store.conversations.runtimes(result["id"]), self.runtimes)
        self.adapter.run.assert_not_called()

    def test_invalid_selection_or_incomplete_roles_never_persist(self):
        for changes in (
            {"provider_id": "unknown"},
            {"model_id": "unknown"},
            {"effort": "max"},
            {"service_tier": "fast"},
            {"session_scope": "stale"},
            {"model_id": ""},
            {"effort": ""},
            {"service_tier": ""},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.create(
                    runtimes={
                        **self.runtimes,
                        Role.IMPLEMENTER: replace(self.runtime, **changes),
                    }
                )
        with self.assertRaisesRegex(ValueError, "all role"):
            self.create(runtimes={Role.ASSISTANT: self.runtime})
        self.assertEqual(self.service.list("workspace"), [])
        self.adapter.run.assert_not_called()

    def test_credentials_required_only_for_active_roles(self):
        missing = replace(
            self.runtime, provider_id="missing", session_scope="missing:v1"
        )
        runtimes = {**self.runtimes, Role.IMPLEMENTER: missing}
        self.create(runtimes=runtimes)
        with self.assertRaises(ProviderUnavailable) as raised:
            self.create(mode=AgentMode.ORCHESTRATED, runtimes=runtimes)
        self.assertEqual(raised.exception.provider_ids, ("missing",))
        self.assertEqual(len(self.service.list("workspace")), 1)

    def test_read_only_composition_and_explicit_creation_without_defaults(self):
        result = self.create()
        read_only = ConversationService(lambda _: self.store)
        self.assertEqual(read_only.get("workspace", result["id"]), result)
        with self.assertRaisesRegex(ValueError, "provider registry"):
            read_only.create(
                "workspace",
                mode=AgentMode.SINGLE,
                sandbox=Sandbox.READ_ONLY,
                autonomous=False,
                peer_workspace=None,
            )
        self.service = ConversationService(
            lambda _: self.store, providers=self.providers
        )
        with self.assertRaisesRegex(ValueError, "configured role runtimes"):
            self.create()
        self.assertNotEqual(self.create(runtimes=self.runtimes)["id"], result["id"])

    def test_default_mapping_is_copied_and_explicit_runtime_override_is_persisted(self):
        self.runtimes.clear()
        first = self.create()
        override = {role: replace(self.runtime, effort="low") for role in Role}
        second = self.create(runtimes=override)
        self.assertEqual(first["codex_runtime"]["assistant"]["effort"], "high")
        self.assertEqual(second["codex_runtime"]["assistant"]["effort"], "low")

    def test_list_matches_legacy_fields_and_order_without_history(self):
        for identity in ("b", "a", "newest"):
            self.store.conversations.create(identity, runtimes=self.runtimes)
        with self.database.connect(write=True) as connection:
            connection.execute("UPDATE conversations SET updated_at = 10")
            connection.execute(
                "UPDATE conversations SET updated_at = 20 WHERE id='newest'"
            )
        result = self.service.list("workspace")
        self.assertEqual([row["id"] for row in result], ["newest", "a", "b"])
        self.assertEqual(set(result[0]), {"id", "title", "naming_state", "updated_at"})

    def test_get_pagination_keeps_stable_message_ids(self):
        result = self.create()
        ids = [
            self.store.conversations.append(result["id"], "user", str(i))
            for i in range(4)
        ]
        page = self.service.get("workspace", result["id"], limit=2, before=3)
        self.assertEqual([message["id"] for message in page["messages"]], ids[1:3])
        self.assertEqual(
            page["message_page"],
            {"start_index": 1, "end_index": 3, "total_messages": 4, "has_more": True},
        )
        with self.assertRaises(ValueError):
            self.service.get("workspace", result["id"], before=1)
