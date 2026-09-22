import json
import unittest

from vibesim_agent.runtime.invocation import RoleContext
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from vibesim_agent.bootstrap import configuration
from vibesim_agent.composition import build_builtin_setup
from vibesim_agent.domain.roles import Role
from vibesim_agent.prompts.render import Prompts
from vibesim_agent.providers.base import AgentRequest
from vibesim_agent.providers.builtin import session_scope
from vibesim_agent.providers.claude.home import ClaudeProfile
from vibesim_agent.runtime.command import ExecutionEnvironment
from vibesim_agent.runtime.homes import role_home


class NamedProviderCompositionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        for name in (".codex", ".codexs"):
            directory = self.root / name
            directory.mkdir()
            (directory / "config.toml").write_text('model_provider="openai"\n')
            (directory / "auth.json").write_text('{"token":"fixture"}')
        self.oauth = self.root / ".claudeme"
        self.oauth.mkdir()
        (self.oauth / ".credentials.json").write_text('{"token":"oauth-original"}')
        (self.oauth / "history.jsonl").write_text("personal-history")
        self.document = {
            "version": 1,
            "providers": {
                "gpt": {
                    "adapter": "codex",
                    "home": str(self.root / ".codex"),
                    "default_model": "gpt-5.6-sol",
                    "default_effort": "high",
                    "models": {"gpt-5.6-sol": {"efforts": ["high"]}},
                },
                "codexs": {
                    "adapter": "codex",
                    "home": str(self.root / ".codexs"),
                    "default_model": "gpt-6-astra",
                    "default_effort": "high",
                    "models": {"gpt-6-astra": {"efforts": ["high", "xhigh"]}},
                    "environment": {"OPENAI_API_KEY": "WORK_CODEX_KEY"},
                },
                "claude": {
                    "adapter": "claude",
                    "default_model": "claude-sonnet-5",
                    "default_effort": "high",
                    "models": {"claude-sonnet-5": {"efforts": ["high"]}},
                    "base_url": "http://fixture-gateway",
                    "environment": {"ANTHROPIC_AUTH_TOKEN": "WORK_CLAUDE_TOKEN"},
                },
                "claudek": {
                    "adapter": "claude",
                    "default_model": "claude-sonnet-5",
                    "default_effort": "high",
                    "models": {"claude-sonnet-5": {"efforts": ["high"]}},
                    "base_url": "http://fixture-other",
                    "environment": {"ANTHROPIC_AUTH_TOKEN": "OTHER_CLAUDE_TOKEN"},
                },
                "claudeme": {
                    "adapter": "claude",
                    "default_model": "claude-sonnet-5",
                    "default_effort": "high",
                    "models": {"claude-sonnet-5": {"efforts": ["high"]}},
                    "home": str(self.oauth),
                },
            },
            "defaults": {
                "orchestrator": "claude",
                "implementer": "codexs",
                "assistant": "claudeme",
            },
        }
        self.path = self.root / "providers.yaml"
        self.env = {
            "HOME": str(self.root),
            "PATH": "/usr/bin",
            "VIBESIM_AGENT_PROVIDERS_FILE": str(self.path),
            "WORK_CLAUDE_TOKEN": "work-secret",
            "OTHER_CLAUDE_TOKEN": "other-secret",
            "WORK_CODEX_KEY": "codex-secret",
            "ANTHROPIC_API_KEY": "unselected-secret",
        }
        self.prompts = Prompts.prepare(self.root / "prompts")

    def settings(self):
        self.path.write_text(yaml.safe_dump(self.document))
        return configuration(environment=self.env, repo_root=self.root)

    def build(self, settings=None):
        settings = settings or self.settings()
        return build_builtin_setup(
            settings,
            lambda request: role_home(
                self.root / "runtime" / request.conversation_id,
                "/runtime",
                request.role,
                request.selection.session_scope,
            ),
            ExecutionEnvironment(
                settings.container, settings.agent, "lock", "/context.json"
            ),
            self.prompts,
            host_environment=self.env,
        )

    def test_connections_use_independent_credentials_defaults_and_homes(self):
        setup = self.build()
        self.assertEqual(
            {
                role.value: runtime.provider_id
                for role, runtime in setup.defaults.items()
            },
            self.document["defaults"],
        )
        expected = {
            "claude": {
                "ANTHROPIC_AUTH_TOKEN": "work-secret",
                "ANTHROPIC_BASE_URL": "http://fixture-gateway",
            },
            "claudek": {
                "ANTHROPIC_AUTH_TOKEN": "other-secret",
                "ANTHROPIC_BASE_URL": "http://fixture-other",
            },
            "codexs": {"OPENAI_API_KEY": "codex-secret"},
            "claudeme": {},
            "gpt": {},
        }
        homes = set()
        for provider_id, variables in expected.items():
            provider = setup.registry.provider(provider_id)
            self.assertTrue(setup.registry.available(provider_id))
            environment = provider.adapter.process_environment
            for key in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY"):
                self.assertEqual(environment.get(key), variables.get(key))
            for key in (
                "WORK_CLAUDE_TOKEN",
                "OTHER_CLAUDE_TOKEN",
                "WORK_CODEX_KEY",
                "ANTHROPIC_API_KEY",
            ):
                self.assertNotIn(key, environment)
                self.assertNotIn(key, setup.docker_environment)
            request = AgentRequest(
                "w",
                "same-conversation",
                "turn",
                Role.ASSISTANT,
                "fixture",
                "container",
                setup.registry.select(provider_id),
            )
            home = provider.adapter.home(request)
            setup.runtimes[provider_id].prepare(home, RoleContext(skills="/workspace/skills"))
            homes.add(home.host)
        self.assertEqual(len(homes), 5)
        self.assertNotIn("secret", json.dumps(setup.registry.catalog()))

    def test_missing_connection_secret_does_not_fall_back_to_global_auth(self):
        self.env.pop("OTHER_CLAUDE_TOKEN")
        setup = self.build()
        self.assertFalse(setup.registry.available("claudek"))
        self.assertTrue(setup.registry.available("claude"))
        self.assertNotIn(
            "ANTHROPIC_AUTH_TOKEN",
            setup.registry.provider("claudek").adapter.process_environment,
        )

    def test_named_connections_only_offer_declared_models(self):
        self.document["providers"]["claudek"]["default_model"] = "glm-5.3-fp4"
        self.document["providers"]["claudek"]["default_effort"] = "max"
        self.document["providers"]["claudek"]["models"] = {
            "glm-5.3-fp4": {"efforts": ["high", "max"]}
        }
        setup = self.build()
        self.assertEqual(setup.registry.select("claudek").effort, "max")
        for provider_id, declaration in self.document["providers"].items():
            self.assertEqual(
                [
                    model.model_id
                    for model in setup.registry.provider(provider_id).catalog()
                ],
                list(declaration["models"]),
            )
        with self.assertRaisesRegex(ValueError, "unknown model"):
            setup.registry.select("claudek", "claude-opus-5")

    def test_explicit_model_list_allows_selection_without_changing_session_scope(self):
        original = self.settings()
        self.document["providers"]["claude"]["models"] = {
            "claude-sonnet-5": {"efforts": ["high"]},
            "claude-opus-5": {"efforts": ["high"]},
        }
        self.document["providers"]["codexs"]["models"] = {
            "gpt-6-astra": {"efforts": ["high", "xhigh"]},
            "gpt-5.6-sol": {"efforts": ["high"]},
        }
        # A cached catalog may enrich declarations but cannot add selectable models.
        (self.root / ".codexs/models_cache.json").write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "slug": "gpt-6-astra",
                            "supported_reasoning_levels": [
                                {"effort": "high"},
                                {"effort": "xhigh"},
                            ],
                        },
                        {"slug": "not-declared"},
                    ]
                }
            )
        )
        settings = self.settings()
        setup = self.build(settings)
        self.assertEqual(
            setup.registry.select("claude", "claude-opus-5").model.model_id,
            "claude-opus-5",
        )
        self.assertEqual(
            setup.registry.select("codexs", effort="xhigh").effort, "xhigh"
        )
        with self.assertRaisesRegex(ValueError, "unknown model"):
            setup.registry.select("codexs", "not-declared")
        for provider_id, adapter in (("claude", "claude"), ("codexs", "codex")):
            self.assertEqual(
                session_scope(settings, provider_id, adapter),
                session_scope(original, provider_id, adapter),
            )

    def test_credentials_and_labels_do_not_change_session_scope(self):
        scope = session_scope(self.settings(), "claudek", "claude")
        self.env["OTHER_CLAUDE_TOKEN"] = "rotated-secret"
        self.document["providers"]["claudek"]["label"] = "New label"
        self.assertEqual(session_scope(self.settings(), "claudek", "claude"), scope)
        self.document["providers"]["claudek"]["session_identity"] = "different-account"
        self.assertNotEqual(session_scope(self.settings(), "claudek", "claude"), scope)

    def test_oauth_copy_preserves_cli_refresh_and_never_copies_history(self):
        home = role_home(
            self.root / "oauth-runtime", "/runtime", Role.ASSISTANT, "scope"
        )
        profile = ClaudeProfile(self.oauth)
        profile.prepare(home, RoleContext(skills="/workspace/skills"))
        target = home.host / ".credentials.json"
        self.assertEqual(target.read_text(), '{"token":"oauth-original"}')
        self.assertFalse((home.host / "history.jsonl").exists())
        target.write_text('{"token":"cli-refreshed"}')
        profile.prepare(home, RoleContext(skills="/workspace/skills"))
        self.assertEqual(target.read_text(), '{"token":"cli-refreshed"}')
        (self.oauth / ".credentials.json").write_text('{"token":"new-host-login"}')
        profile.prepare(home, RoleContext(skills="/workspace/skills"))
        self.assertEqual(target.read_text(), '{"token":"new-host-login"}')
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
