"""Built-in capabilities and credential-free backend compatibility identities."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from vibesim_agent.providers.base import OutputMode
from vibesim_agent.providers.builtin import (
    CLAUDE_ENVIRONMENT,
    GPT_MODELS,
    build_registry,
    guarded_profile_prepare,
    legacy_model_aliases,
    provider_environments,
    session_scope,
)
from vibesim_agent.runtime.invocation import InvocationHome
from vibesim_agent.settings import ConfigurationError, load_settings


class BuiltinProviderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.declarations = provider_environments(self.root)
        self.adapters = {
            name: Mock(adapter_id=adapter)
            for name, adapter in (
                ("gpt", "codex"),
                ("deepseek", "codex"),
                ("claude", "claude"),
            )
        }
        for name in (".codex", ".codex-ds"):
            home = self.root / name
            home.mkdir()
            (home / "config.toml").write_text(
                'model_provider="custom"\n[model_providers.custom]\nbase_url="http://backend/v1"\nwire_api="responses"\n'
            )
        self.settings = self.settings_for({})

    def settings_for(self, environment):
        return load_settings(
            environment=environment, repo_root=self.root, providers=self.declarations
        )

    def build(self, settings=None):
        return build_registry(settings or self.settings, adapters=self.adapters)

    def test_default_declarations_and_fallback_models_match_legacy(self):
        self.assertEqual(
            [declaration.provider_id for declaration in self.declarations],
            ["gpt", "deepseek", "claude"],
        )
        self.assertEqual(self.declarations[2].secret_names, CLAUDE_ENVIRONMENT)
        registry = self.build()
        self.assertEqual(
            [model.model_id for model in registry.provider("gpt").catalog()],
            list(GPT_MODELS),
        )
        self.assertEqual(registry.select("gpt").effort, "xhigh")
        deepseek = registry.select("deepseek")
        self.assertEqual(deepseek.model.model_id, "deepseek-ai/DeepSeek-V4-Flash-0731")
        self.assertEqual(deepseek.model.efforts, ("high", "xhigh", "max"))
        self.assertEqual(deepseek.model.output_mode, OutputMode.PROMPT)
        self.assertTrue(deepseek.model.resumable)
        self.assertEqual(
            [model.label for model in registry.provider("claude").catalog()],
            ["Claude Sonnet 5", "Claude Opus 5"],
        )
        for model in registry.provider("claude").catalog():
            self.assertEqual(model.efforts, ("low", "medium", "high", "xhigh", "max"))
            self.assertEqual(model.output_mode, OutputMode.STRUCTURED)
            self.assertEqual(model.service_tiers, ("default",))
        self.assertIs(registry.provider("gpt").adapter, self.adapters["gpt"])
        self.assertIs(registry.provider("deepseek").adapter, self.adapters["deepseek"])
        self.assertIsNot(self.adapters["gpt"], self.adapters["deepseek"])

    def test_credentials_match_legacy_and_backend_url_is_not_authentication(self):
        registry = self.build()
        self.assertTrue(registry.available("gpt"))
        self.assertFalse(registry.available("deepseek"))
        self.assertFalse(registry.available("claude"))
        for token in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
        ):
            authenticated = self.build(
                self.settings_for({token: "secret", "VLLM_API_KEY": "secret"})
            )
            self.assertTrue(authenticated.available("claude"))
            self.assertTrue(authenticated.available("deepseek"))
            self.assertNotIn("secret", json.dumps(authenticated.catalog()))
        self.assertFalse(
            self.build(
                self.settings_for({"ANTHROPIC_BASE_URL": "http://backend"})
            ).available("claude")
        )
        (self.root / ".codex/config.toml").unlink()
        self.assertFalse(self.build().available("gpt"))

    def test_catalog_precedence_allowlist_refresh_and_configured_effort_priority(self):
        cache = self.root / ".codex/models_cache.json"
        alternate = self.root / ".codex/models_catalog.json"
        cache.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "slug": "gpt-5.6-sol",
                            "display_name": "Preferred",
                            "supported_reasoning_levels": [
                                {"effort": "low"},
                                {"effort": "max"},
                            ],
                            "default_reasoning_level": "low",
                            "additional_speed_tiers": ["fast"],
                        },
                        {"slug": "hidden-model"},
                    ]
                }
            )
        )
        alternate.write_text(
            json.dumps(
                {"models": [{"slug": "gpt-5.6-sol", "display_name": "Alternate"}]}
            )
        )
        registry = self.build(self.settings_for({"VIBESIM_PROVIDER_GPT_EFFORT": "max"}))
        model = registry.select("gpt").model
        self.assertEqual(
            (model.label, model.default_effort, model.service_tiers),
            ("Preferred", "max", ("default", "fast")),
        )
        self.assertEqual(
            registry.select("gpt", service_tier="fast").service_tier, "fast"
        )
        self.assertEqual(len(registry.provider("gpt").catalog()), 3)
        registry = self.build(
            self.settings_for({"VIBESIM_PROVIDER_GPT_EFFORT": "unsupported"})
        )
        self.assertEqual(registry.select("gpt").model.default_effort, "low")
        cache.unlink()
        self.assertEqual(registry.select("gpt").model.label, "Alternate")
        alternate.unlink()
        self.assertEqual(registry.select("gpt").model.default_effort, "xhigh")

    def test_deepseek_catalog_order_custom_default_and_claude_aliases(self):
        settings = self.settings_for(
            {
                "VIBESIM_PROVIDER_DEEPSEEK_MODEL": "custom-ds",
                "VIBESIM_PROVIDER_CLAUDE_MODEL": "opus",
            }
        )
        for name, label in (
            ("models_catalog.json", "Preferred"),
            ("models_cache.json", "Other"),
        ):
            (self.root / ".codex-ds" / name).write_text(
                json.dumps({"models": [{"slug": "custom-ds", "display_name": label}]})
            )
        registry = self.build(settings)
        self.assertEqual(registry.select("deepseek").model.label, "Preferred")
        self.assertEqual(registry.select("claude").model.model_id, "claude-opus-5")
        self.assertEqual(
            legacy_model_aliases(registry),
            {
                "traditional": "gpt-5.6-sol",
                "codexds": "custom-ds",
                "sonnet": "claude-sonnet-5",
                "opus": "claude-opus-5",
            },
        )
        limited = settings.model_copy(
            update={"providers": {"gpt": settings.providers["gpt"]}}
        )
        self.assertEqual(
            legacy_model_aliases(self.build(limited)), {"traditional": "gpt-5.6-sol"}
        )

    def test_scope_tracks_backend_and_adapter_but_not_credentials_or_model(self):
        original = session_scope(self.settings, "gpt", "codex")
        changed = self.settings_for(
            {
                "VIBESIM_PROVIDER_GPT_MODEL": "gpt-5.6-luna",
                "VIBESIM_PROVIDER_GPT_EFFORT": "low",
                "VLLM_API_KEY": "rotated",
            }
        )
        self.assertEqual(session_scope(changed, "gpt", "codex"), original)
        config = self.root / ".codex/config.toml"
        text = config.read_text()
        config.write_text(text + 'experimental_bearer_token="rotated-secret"\n')
        self.assertEqual(session_scope(self.settings, "gpt", "codex"), original)
        config.write_text(text.replace("http://backend/v1", "http://other/v1"))
        self.assertNotEqual(session_scope(self.settings, "gpt", "codex"), original)
        self.assertNotIn("http://", session_scope(self.settings, "gpt", "codex"))
        self.assertNotEqual(session_scope(self.settings, "gpt", "claude"), original)
        self.assertNotEqual(session_scope(self.settings, "deepseek", "codex"), original)
        claude = session_scope(self.settings, "claude", "claude")
        self.assertEqual(
            session_scope(
                self.settings_for({"ANTHROPIC_API_KEY": "rotated"}), "claude", "claude"
            ),
            claude,
        )
        self.assertNotEqual(
            session_scope(
                self.settings_for({"ANTHROPIC_BASE_URL": "http://other"}),
                "claude",
                "claude",
            ),
            claude,
        )

    def test_effective_codex_profile_and_invalid_config_are_explicit(self):
        config = self.root / ".codex/config.toml"
        before = session_scope(self.settings, "gpt", "codex")
        config.write_text(
            config.read_text() + '\n[profiles.other]\nmodel_provider="alternate"\n'
        )
        self.assertEqual(session_scope(self.settings, "gpt", "codex"), before)
        config.write_text('profile="other"\n' + config.read_text())
        self.assertNotEqual(session_scope(self.settings, "gpt", "codex"), before)
        config.write_text('profile="missing"\n')
        with self.assertRaises(ConfigurationError):
            self.build()
        config.write_text("invalid=")
        with self.assertRaisesRegex(ConfigurationError, "cannot read"):
            self.build()

    def test_guard_rejects_changed_backend_before_or_during_copy(self):
        provider = self.build().provider("gpt")
        config = self.root / ".codex/config.toml"
        original = config.read_text()
        prepare = Mock()
        guarded = guarded_profile_prepare(self.settings, provider, prepare)
        home = InvocationHome(self.root / "runtime", "/runtime")
        config.write_text(original.replace("http://backend", "http://other"))
        with self.assertRaisesRegex(ConfigurationError, "rebuild"):
            guarded(home)
        prepare.assert_not_called()
        config.write_text(original)
        prepare.side_effect = lambda _: config.write_text(
            original.replace("http://backend", "http://other")
        )
        with self.assertRaisesRegex(ConfigurationError, "rebuild"):
            guarded(home)
        prepare.assert_called_once_with(home)

    def test_explicit_inputs_do_not_read_environment_or_launch_adapters(self):
        with (
            patch("os.getenv", side_effect=AssertionError("implicit environment")),
            patch("os.environ", {}),
        ):
            self.assertEqual(len(provider_environments(self.root)), 3)
            self.assertEqual(len(self.build().catalog()), 3)
        for adapter in self.adapters.values():
            adapter.run.assert_not_called()
        with self.assertRaises(ConfigurationError):
            build_registry(self.settings, adapters={})
        with self.assertRaises(ConfigurationError):
            provider_environments(Path("relative"))
        with self.assertRaisesRegex(ValueError, "default model absent"):
            self.build(
                self.settings_for({"VIBESIM_PROVIDER_GPT_MODEL": "unlisted-model"})
            )
