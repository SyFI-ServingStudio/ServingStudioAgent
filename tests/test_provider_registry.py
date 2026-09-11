"""Provider extension, model capabilities, credential and resume boundaries."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import SecretStr
from vibesim_agent.domain.roles import Role
from vibesim_agent.providers.base import (
    AgentRequest,
    Credentials,
    Model,
    OutputMode,
    Provider,
)
from vibesim_agent.providers.catalog import FileCatalog
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.settings import ProviderSettings


class RecordingAdapter:
    adapter_id = "test"

    def __init__(self):
        self.calls = []

    async def run(self, request):
        self.calls.append(request)
        yield {"kind": "final", "text": "answer"}


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_direct_provider_registration_requires_environment_compatible_identity(self):
        model = Model("model", "Model", ("high",), "high")
        registry = ProviderRegistry()
        for identity in ("Example", "example-name", "0example", "", " example "):
            with self.subTest(identity=identity), self.assertRaisesRegex(
                ValueError, "provider_id"
            ):
                registry.register(Provider(
                    identity, "Example", RecordingAdapter(),
                    ProviderSettings(model="model", effort="high"),
                    "example:v1", lambda: (model,),
                ))
        self.assertEqual(registry.catalog(), [])

    def test_unsupported_configured_and_explicit_efforts_reject(self):
        model = Model("model", "Model", ("low", "high"), "high")
        registry = ProviderRegistry()
        registry.register(Provider(
            "example", "Example", RecordingAdapter(),
            ProviderSettings(model="model", effort="max"),
            "example:v1", lambda: (model,),
        ))
        self.assertEqual(registry.select("example", effort="low").effort, "low")
        for effort in (None, "max"):
            with self.subTest(effort=effort), self.assertRaisesRegex(
                ValueError, "unsupported effort"
            ):
                registry.select("example", effort=effort)

    async def test_two_profiles_share_adapter_but_preserve_capability_and_scope(self):
        adapter = RecordingAdapter()
        structured = Model("same-model", "Model", ("low", "high"), "high")
        prompt_only = replace(structured, output_mode=OutputMode.PROMPT)
        registry = ProviderRegistry()
        for identity, model in (("first", structured), ("second", prompt_only)):
            registry.register(
                Provider(
                    identity,
                    identity,
                    adapter,
                    ProviderSettings(model=model.model_id, effort="high"),
                    f"test:{identity}:v1",
                    lambda m=model: (m,),
                )
            )
        for identity, expected in (("first", True), ("second", False)):
            request = AgentRequest(
                "w",
                "c",
                "t",
                Role.ASSISTANT,
                "question",
                "container",
                registry.select(identity),
                output_schema=Path("/schema.json"),
            )
            events = [event async for event in registry.run(request)]
            self.assertEqual(events, [{"kind": "final", "text": "answer"}])
            self.assertEqual(adapter.calls[-1].structured_output, expected)
        self.assertNotEqual(
            adapter.calls[0].selection.session_scope,
            adapter.calls[1].selection.session_scope,
        )
        self.assertEqual([p["id"] for p in registry.catalog()], ["first", "second"])

    async def test_invalid_selection_and_missing_auth_never_launch_adapter(self):
        adapter = RecordingAdapter()
        model = Model("model", "Model", ("high",), "high")
        provider = Provider(
            "custom",
            "Custom",
            adapter,
            ProviderSettings(model="model", effort="high"),
            "custom:v1",
            lambda: (model,),
            Credentials(any_secrets=("TOKEN", "OTHER_TOKEN")),
        )
        registry = ProviderRegistry()
        registry.register(provider)
        with self.assertRaisesRegex(ValueError, "unsupported effort"):
            registry.select("custom", effort="max")
        with self.assertRaisesRegex(ValueError, "unsupported service tier"):
            registry.select("custom", service_tier="fast")
        request = AgentRequest(
            "w", "c", "t", Role.ASSISTANT, "q", "container", registry.select("custom")
        )
        with self.assertRaisesRegex(ValueError, "credentials unavailable"):
            _ = [event async for event in registry.run(request)]
        self.assertEqual(adapter.calls, [])
        authenticated = ProviderRegistry({"OTHER_TOKEN": SecretStr("credential")})
        authenticated.register(provider)
        self.assertTrue(authenticated.available("custom"))
        self.assertNotIn("credential", json.dumps(authenticated.catalog()))

    async def test_nonresumable_model_is_rejected_before_launch(self):
        adapter = RecordingAdapter()
        model = Model("model", "Model", ("high",), "high", resumable=False)
        registry = ProviderRegistry()
        registry.register(
            Provider(
                "custom",
                "Custom",
                adapter,
                ProviderSettings(model="model", effort="high"),
                "custom:v1",
                lambda: (model,),
            )
        )
        request = AgentRequest(
            "w",
            "c",
            "t",
            Role.ASSISTANT,
            "q",
            "container",
            registry.select("custom"),
            session_id="old-session",
        )
        with self.assertRaisesRegex(ValueError, "does not support session resume"):
            _ = [event async for event in registry.run(request)]
        self.assertEqual(adapter.calls, [])

    async def test_stale_capabilities_or_forged_session_scope_never_launch(self):
        adapter = RecordingAdapter()
        models = [Model("model", "Model", ("high",), "high")]
        registry = ProviderRegistry()
        registry.register(Provider(
            "custom", "Custom", adapter,
            ProviderSettings(model="model", effort="high"), "custom:v1",
            lambda: tuple(models),
        ))
        request = AgentRequest(
            "w", "c", "t", Role.ASSISTANT, "q", "container",
            registry.select("custom"), session_id="existing-session",
        )
        forged = replace(request, selection=replace(
            request.selection, session_scope="incompatible:v2"
        ))
        with self.assertRaisesRegex(ValueError, "session scope changed"):
            _ = [event async for event in registry.run(forged)]
        models[0] = replace(models[0], output_mode=OutputMode.PROMPT)
        with self.assertRaisesRegex(ValueError, "model capabilities"):
            _ = [event async for event in registry.run(request)]
        self.assertEqual(adapter.calls, [])
        refreshed = replace(request, selection=registry.select("custom"))
        _ = [event async for event in registry.run(refreshed)]
        self.assertEqual(adapter.calls, [refreshed])

    def test_catalog_refresh_preserves_yaml_efforts_and_does_not_expose_unlisted_models(
        self,
    ):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            base = Model("allowed", "Allowed", ("low", "high"), "high")
            catalog = FileCatalog((base,), (path,))
            self.assertEqual(catalog(), (base,))
            path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "slug": "allowed",
                                "supported_reasoning_levels": [{"effort": "max"}],
                                "additional_speed_tiers": ["fast"],
                            },
                            {"slug": "hidden"},
                        ]
                    }
                )
            )
            updated = catalog()
            self.assertEqual(len(updated), 1)
            self.assertEqual(updated[0].efforts, ("low", "high"))
            self.assertEqual(updated[0].default_effort, "high")
            self.assertEqual(updated[0].service_tiers, ("default", "fast"))
            path.write_text("invalid")
            self.assertEqual(catalog(), (base,))
