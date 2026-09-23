"""Configured provider capabilities, credentials, and session identities."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from vibesim_agent.api.catalog import _environment_names
from vibesim_agent.providers.builtin import (
    build_registry,
    guarded_profile_prepare,
    legacy_model_aliases,
    provider_environment,
    session_scope,
)
from vibesim_agent.runtime.invocation import InvocationHome, RoleContext
from vibesim_agent.settings import ConfigurationError, load_settings


class ConfiguredProviderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.codex_home = self.root / ".codex"
        self.codex_home.mkdir()
        (self.codex_home / "config.toml").write_text(
            'model_provider="custom"\n[model_providers.custom]\n'
            'base_url="http://backend/v1"\nwire_api="responses"\n'
        )
        self.document = {
            "version": 1,
            "providers": {
                "gpt": {
                    "adapter": "codex",
                    "home": str(self.codex_home),
                    "default_model": "gpt-primary",
                    "default_effort": "high",
                    "models": {
                        "gpt-primary": {"efforts": ["low", "high", "ultra"]},
                        "gpt-fast": {"efforts": ["low", "high"]},
                    },
                },
                "claude": {
                    "adapter": "claude",
                    "base_url": "https://claude.example.test",
                    "environment": {"ANTHROPIC_AUTH_TOKEN": "CLAUDE_TOKEN"},
                    "default_model": "claude-primary",
                    "default_effort": "high",
                    "models": {
                        "claude-primary": {"efforts": ["low", "high", "max"]}
                    },
                },
            },
            "defaults": {
                "orchestrator": "gpt",
                "implementer": "gpt",
                "assistant": "claude",
            },
        }
        (self.root / "providers.yaml").write_text(json.dumps(self.document))
        self.adapters = {
            "gpt": Mock(adapter_id="codex"),
            "claude": Mock(adapter_id="claude"),
        }

    def settings(self, **environment):
        return load_settings(
            repo_root=self.root,
            environment={"HOME": str(self.root), **environment},
        )

    def registry(self, settings=None):
        return build_registry(settings or self.settings(), adapters=self.adapters)

    def test_yaml_models_and_efforts_are_authoritative(self):
        registry = self.registry()
        catalog = {entry["id"]: entry for entry in registry.catalog()}
        self.assertEqual(
            [model["id"] for model in catalog["gpt"]["models"]],
            ["gpt-primary", "gpt-fast"],
        )
        self.assertEqual(
            catalog["gpt"]["models"][0]["efforts"],
            ["low", "high", "ultra"],
        )
        self.assertEqual(
            catalog["claude"]["models"][0]["efforts"],
            ["low", "high", "max"],
        )
        self.assertEqual(registry.select("gpt").model.model_id, "gpt-primary")
        self.assertEqual(registry.select("gpt").effort, "high")

    def test_cached_catalog_cannot_change_models_or_efforts(self):
        (self.codex_home / "models_cache.json").write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "slug": "gpt-primary",
                            "display_name": "Catalog label",
                            "supported_reasoning_levels": [{"effort": "medium"}],
                            "additional_speed_tiers": ["fast"],
                        },
                        {"slug": "hidden"},
                    ]
                }
            )
        )
        [primary, fast] = self.registry().provider("gpt").catalog()
        self.assertEqual(primary.label, "Catalog label")
        self.assertEqual(primary.efforts, ("low", "high", "ultra"))
        self.assertEqual(primary.service_tiers, ("default", "fast"))
        self.assertEqual(fast.model_id, "gpt-fast")

    def test_credentials_are_explicit_and_never_enter_catalog(self):
        unavailable = self.registry()
        self.assertFalse(unavailable.available("claude"))
        available = self.registry(self.settings(CLAUDE_TOKEN="private-token"))
        self.assertTrue(available.available("claude"))
        self.assertEqual(
            provider_environment(self.settings(CLAUDE_TOKEN="private-token"), "claude"),
            {
                "ANTHROPIC_AUTH_TOKEN": "private-token",
                "ANTHROPIC_BASE_URL": "https://claude.example.test",
            },
        )
        self.assertNotIn("private-token", json.dumps(available.catalog()))

    def test_inline_secret_references_are_not_browser_environment_names(self):
        self.assertEqual(
            _environment_names(
                ("WORK_CLAUDE_TOKEN", "inline:claudek:ANTHROPIC_AUTH_TOKEN")
            ),
            ["WORK_CLAUDE_TOKEN"],
        )

    def test_scope_tracks_backend_not_credentials(self):
        settings = self.settings(CLAUDE_TOKEN="one")
        original = session_scope(settings, "gpt", "codex")
        rotated = self.settings(CLAUDE_TOKEN="two")
        self.assertEqual(session_scope(rotated, "gpt", "codex"), original)
        config = self.codex_home / "config.toml"
        config.write_text(config.read_text().replace("http://backend", "http://other"))
        self.assertNotEqual(session_scope(settings, "gpt", "codex"), original)

    def test_guard_and_adapter_mismatch_fail_before_execution(self):
        settings = self.settings()
        provider = self.registry(settings).provider("gpt")
        prepare = Mock()
        guarded = guarded_profile_prepare(settings, provider, prepare)
        config = self.codex_home / "config.toml"
        config.write_text(config.read_text().replace("http://backend", "http://other"))
        with self.assertRaisesRegex(ConfigurationError, "rebuild"):
            guarded(
                InvocationHome(self.root / "runtime", "/runtime"),
                RoleContext(skills="/workspace/skills"),
            )
        prepare.assert_not_called()
        with self.assertRaises(ConfigurationError):
            build_registry(settings, adapters={})

    def test_legacy_message_aliases_do_not_add_models(self):
        registry = self.registry()
        self.assertEqual(legacy_model_aliases(registry), {"traditional": "gpt-primary"})
        self.assertNotIn("sonnet", legacy_model_aliases(registry))


if __name__ == "__main__":
    unittest.main()
