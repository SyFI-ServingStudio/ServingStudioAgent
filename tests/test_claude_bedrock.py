"""Claude connections through Amazon Bedrock: config, credentials and isolation."""

import configparser
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import yaml

from tests.test_claude_command import ClaudeCommandTests
from vibesim_agent.bootstrap import configuration
from vibesim_agent.composition import build_builtin_setup
from vibesim_agent.domain.roles import Role
from vibesim_agent.prompts.render import Prompts
from vibesim_agent.provider_config import load_provider_config
from vibesim_agent.providers.base import AgentRequest
from vibesim_agent.providers.builtin import (
    build_registry,
    provider_environment,
    session_scope,
)
from vibesim_agent.providers.claude.home import ClaudeProfile
from vibesim_agent.runtime.command import ExecutionEnvironment
from vibesim_agent.runtime.homes import role_home
from vibesim_agent.runtime.invocation import InvocationHome, RoleContext
from vibesim_agent.settings import ConfigurationError, load_settings

MODEL = "us.anthropic.claude-opus-5-5"
CREDENTIALS = (
    "[default]\n"
    "aws_access_key_id = AKIDEFAULT\n"
    "aws_secret_access_key = default-secret\n"
    "region = us-west-2\n"
    "[other]\n"
    "aws_access_key_id = AKIDOTHER\n"
    "aws_secret_access_key = other-secret\n"
)


def bedrock_connection(**overrides):
    connection = {
        "adapter": "claude",
        "label": "Claude (Bedrock)",
        "bedrock": {"region": "us-east-1", "profile": "default"},
        "default_model": MODEL,
        "default_effort": "high",
        "models": {MODEL: {"efforts": ["low", "high", "max"]}},
    }
    connection.update(overrides)
    return connection


class BedrockFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        (self.root / ".aws").mkdir()
        self.credentials = self.root / ".aws/credentials"
        self.credentials.write_text(CREDENTIALS)
        self.path = self.root / "providers.yaml"

    def document(self, **connections):
        connections = connections or {"bedrock": bedrock_connection()}
        first = next(iter(connections))
        return {
            "version": 1,
            "providers": connections,
            "defaults": {role: first for role in ("orchestrator", "implementer", "assistant")},
        }

    def load(self, document=None, **environment):
        self.path.write_text(json.dumps(document or self.document()))
        return load_provider_config(
            self.path, environment={"HOME": str(self.root), **environment}
        )

    def settings(self, document=None, **environment):
        self.path.write_text(json.dumps(document or self.document()))
        return load_settings(
            repo_root=self.root, environment={"HOME": str(self.root), **environment}
        )


class BedrockConfigTests(BedrockFixture):
    def test_profile_mode_defaults_to_the_home_credentials_file(self):
        bedrock = self.load().connections["bedrock"].bedrock
        self.assertEqual(bedrock.region, "us-east-1")
        self.assertEqual(bedrock.profile, "default")
        self.assertEqual(bedrock.credentials_file, self.credentials)

    def test_explicit_credentials_file_expands_home(self):
        connection = bedrock_connection(
            bedrock={
                "region": "eu-central-1",
                "profile": "other",
                "credentials_file": "~/elsewhere/credentials",
            }
        )
        bedrock = self.load(self.document(bedrock=connection)).connections[
            "bedrock"
        ].bedrock
        self.assertEqual(bedrock.credentials_file, self.root / "elsewhere/credentials")

    def test_environment_mode_accepts_api_key_or_key_pair(self):
        for refs in (
            {"AWS_BEARER_TOKEN_BEDROCK": "BEDROCK_KEY"},
            {"AWS_ACCESS_KEY_ID": "KEY_ID", "AWS_SECRET_ACCESS_KEY": "SECRET"},
            {
                "AWS_ACCESS_KEY_ID": "KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "SECRET",
                "AWS_SESSION_TOKEN": "SESSION",
            },
        ):
            with self.subTest(refs=refs):
                connection = bedrock_connection(
                    bedrock={"region": "us-east-1"}, environment=refs
                )
                loaded = self.load(self.document(bedrock=connection))
                self.assertEqual(loaded.connections["bedrock"].environment, refs)
                self.assertIsNone(loaded.connections["bedrock"].bedrock.profile)

    def test_invalid_bedrock_declarations_are_rejected(self):
        region = {"region": "us-east-1"}
        cases = {
            "no credential source": bedrock_connection(bedrock=region),
            "profile and environment": bedrock_connection(
                environment={"AWS_BEARER_TOKEN_BEDROCK": "KEY"}
            ),
            "anthropic credential": bedrock_connection(
                bedrock=region, environment={"ANTHROPIC_API_KEY": "KEY"}
            ),
            "incomplete key pair": bedrock_connection(
                bedrock=region, environment={"AWS_ACCESS_KEY_ID": "KEY_ID"}
            ),
            "mixed auth": bedrock_connection(
                bedrock=region,
                environment={
                    "AWS_BEARER_TOKEN_BEDROCK": "KEY",
                    "AWS_ACCESS_KEY_ID": "KEY_ID",
                    "AWS_SECRET_ACCESS_KEY": "SECRET",
                },
            ),
            "gateway endpoint": bedrock_connection(base_url="https://gateway.test"),
            "claude login home": bedrock_connection(home="~/.claude"),
            "codex adapter": bedrock_connection(adapter="codex", home="~/.codex"),
            "missing region": bedrock_connection(bedrock={"profile": "default"}),
            "bad region": bedrock_connection(
                bedrock={"region": "US_EAST", "profile": "default"}
            ),
            "unknown field": bedrock_connection(
                bedrock={"region": "us-east-1", "profile": "default", "role": "x"}
            ),
            "file without profile": bedrock_connection(
                bedrock={**region, "credentials_file": "/creds"},
                environment={"AWS_BEARER_TOKEN_BEDROCK": "KEY"},
            ),
            "relative file": bedrock_connection(
                bedrock={**region, "profile": "default", "credentials_file": "creds"}
            ),
            "not a mapping": bedrock_connection(bedrock="us-east-1"),
        }
        for name, connection in cases.items():
            with self.subTest(name):
                with self.assertRaises(ConfigurationError):
                    self.load(self.document(bedrock=connection))


class BedrockRegistryTests(BedrockFixture):
    def test_environment_selects_bedrock_region_and_background_model(self):
        connection = bedrock_connection(
            bedrock={
                "region": "us-east-1",
                "profile": "default",
                "small_fast_model": "us.anthropic.claude-haiku-4-5",
            }
        )
        settings = self.settings(self.document(bedrock=connection))
        self.assertEqual(
            provider_environment(settings, "bedrock"),
            {
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "AWS_REGION": "us-east-1",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "us.anthropic.claude-haiku-4-5",
            },
        )

    def test_profile_availability_follows_the_credentials_file(self):
        settings = self.settings()
        adapters = {"bedrock": Mock(adapter_id="claude")}
        self.assertTrue(build_registry(settings, adapters=adapters).available("bedrock"))
        self.credentials.unlink()
        self.assertFalse(build_registry(settings, adapters=adapters).available("bedrock"))

    def test_key_pair_requires_every_declared_secret(self):
        connection = bedrock_connection(
            bedrock={"region": "us-east-1"},
            environment={"AWS_ACCESS_KEY_ID": "KEY_ID", "AWS_SECRET_ACCESS_KEY": "SECRET"},
        )
        document = self.document(bedrock=connection)
        adapters = {"bedrock": Mock(adapter_id="claude")}
        partial = self.settings(document, KEY_ID="AKID")
        self.assertFalse(build_registry(partial, adapters=adapters).available("bedrock"))
        complete = self.settings(document, KEY_ID="AKID", SECRET="secret")
        registry = build_registry(complete, adapters=adapters)
        self.assertTrue(registry.available("bedrock"))
        self.assertEqual(
            provider_environment(complete, "bedrock")["AWS_SECRET_ACCESS_KEY"], "secret"
        )
        self.assertNotIn("secret", json.dumps(registry.catalog()))

    def test_scope_tracks_region_and_profile_not_keys(self):
        original = session_scope(self.settings(), "bedrock", "claude")
        self.credentials.write_text(CREDENTIALS.replace("default-secret", "rotated"))
        self.assertEqual(session_scope(self.settings(), "bedrock", "claude"), original)
        for bedrock in (
            {"region": "us-west-2", "profile": "default"},
            {"region": "us-east-1", "profile": "other"},
        ):
            with self.subTest(bedrock=bedrock):
                document = self.document(bedrock=bedrock_connection(bedrock=bedrock))
                self.assertNotEqual(
                    session_scope(self.settings(document), "bedrock", "claude"),
                    original,
                )


class BedrockProfileTests(BedrockFixture):
    def setUp(self):
        super().setUp()
        self.home = InvocationHome(self.root / "role", "/role-home")

    def profile(self, name="default"):
        return ClaudeProfile(aws_credentials=self.credentials, aws_profile=name)

    def test_only_the_selected_profiles_keys_are_copied_privately(self):
        self.profile().prepare(self.home, RoleContext(skills="/workspace/skills"))
        target = self.home.host / ".aws"
        copied = configparser.RawConfigParser()
        copied.read_string((target / "credentials").read_text())
        self.assertEqual(copied.sections(), ["default"])
        self.assertEqual(
            dict(copied["default"]),
            {"aws_access_key_id": "AKIDEFAULT", "aws_secret_access_key": "default-secret"},
        )
        self.assertEqual((target / "config").read_text(), "")
        self.assertEqual(target.stat().st_mode & 0o777, 0o700)
        for name in ("credentials", "config"):
            self.assertEqual((target / name).stat().st_mode & 0o777, 0o600)
        self.assertNotIn("other-secret", (target / "credentials").read_text())

    def test_rotated_host_keys_reach_the_next_prepare(self):
        self.profile().prepare(self.home, RoleContext(skills="/workspace/skills"))
        self.credentials.write_text(CREDENTIALS.replace("default-secret", "rotated"))
        self.profile().prepare(self.home, RoleContext(skills="/workspace/skills"))
        self.assertIn("rotated", (self.home.host / ".aws/credentials").read_text())

    def test_missing_or_keyless_profiles_fail_closed(self):
        self.credentials.write_text(
            CREDENTIALS + "[sso]\nsso_session = corp\n"
        )
        for name in ("absent", "sso"):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    self.profile(name).prepare(self.home, RoleContext(skills="/workspace/skills"))

    def test_symlinked_credential_targets_are_rejected(self):
        (self.home.host / ".aws").mkdir(parents=True)
        (self.home.host / ".aws/credentials").symlink_to(self.root / "elsewhere")
        with self.assertRaises((ValueError, OSError)):
            self.profile().prepare(self.home, RoleContext(skills="/workspace/skills"))
        self.assertFalse((self.root / "elsewhere").exists())

    def test_profile_and_claude_login_are_exclusive(self):
        with self.assertRaises(ValueError):
            ClaudeProfile(self.root, aws_credentials=self.credentials, aws_profile="x")
        with self.assertRaises(ValueError):
            ClaudeProfile(aws_profile="default")


class BedrockCommandTests(unittest.TestCase):
    def test_profile_mode_points_the_sdk_into_the_runtime_home(self):
        fixture = ClaudeCommandTests()
        builder = replace(fixture.builder(), aws_profile="default")
        command = builder.build_tracked(
            fixture.request(), home="/role home", pid_file="/role home/call.pid"
        )
        for value in (
            "AWS_PROFILE=default",
            "AWS_SHARED_CREDENTIALS_FILE=/role home/.aws/credentials",
            "AWS_CONFIG_FILE=/role home/.aws/config",
        ):
            self.assertIn(value, " ".join(command))

    def test_other_connections_set_no_aws_variables(self):
        fixture = ClaudeCommandTests()
        command = fixture.builder().build_tracked(
            fixture.request(), home="/home", pid_file="/home/call.pid"
        )
        self.assertNotIn("AWS_", " ".join(command))


class BedrockCompositionTests(BedrockFixture):
    def setUp(self):
        super().setUp()
        self.env = {
            "HOME": str(self.root),
            "PATH": "/usr/bin",
            "VIBESIM_AGENT_PROVIDERS_FILE": str(self.path),
            "GATEWAY_TOKEN": "gateway-secret",
            # The operator's own Claude Code-on-Bedrock shell settings.
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": "ap-south-1",
            "AWS_PROFILE": "operator",
            "AWS_ACCESS_KEY_ID": "AKIDOPERATOR",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "operator-model",
        }
        gateway = {
            "adapter": "claude",
            "default_model": "claude-sonnet-5",
            "default_effort": "high",
            "models": {"claude-sonnet-5": {"efforts": ["high"]}},
            "base_url": "http://fixture-gateway",
            "environment": {"ANTHROPIC_AUTH_TOKEN": "GATEWAY_TOKEN"},
        }
        self.path.write_text(
            yaml.safe_dump(self.document(bedrock=bedrock_connection(), gateway=gateway))
        )
        self.settings = configuration(environment=self.env, repo_root=self.root)
        self.setup = build_builtin_setup(
            self.settings,
            lambda request: role_home(
                self.root / "runtime" / request.conversation_id,
                "/runtime",
                request.role,
                request.selection.session_scope,
            ),
            ExecutionEnvironment(
                self.settings.container, self.settings.agent, "lock", "/context.json"
            ),
            Prompts.prepare(self.root / "prompts"),
            host_environment=self.env,
        )

    def test_host_bedrock_settings_never_leak_into_any_connection(self):
        bedrock = self.setup.registry.provider("bedrock").adapter.process_environment
        self.assertEqual(bedrock["CLAUDE_CODE_USE_BEDROCK"], "1")
        self.assertEqual(bedrock["AWS_REGION"], "us-east-1")
        gateway = self.setup.registry.provider("gateway").adapter.process_environment
        for environment in (bedrock, gateway, self.setup.docker_environment):
            for key in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "ANTHROPIC_DEFAULT_OPUS_MODEL"):
                self.assertNotIn(key, environment)
        self.assertNotIn("CLAUDE_CODE_USE_BEDROCK", gateway)
        self.assertNotIn("AWS_REGION", gateway)

    def test_prepared_bedrock_home_carries_the_profile_and_command_uses_it(self):
        provider = self.setup.registry.provider("bedrock")
        self.assertTrue(self.setup.registry.available("bedrock"))
        request = AgentRequest(
            "w",
            "conversation",
            "turn",
            Role.ASSISTANT,
            "fixture",
            "container",
            self.setup.registry.select("bedrock"),
        )
        home = provider.adapter.home(request)
        self.setup.runtimes["bedrock"].prepare(home, RoleContext(skills="/workspace/skills"))
        self.assertIn("AKIDEFAULT", (home.host / ".aws/credentials").read_text())
        self.assertEqual(provider.adapter.command.aws_profile, "default")
        self.assertIn("CLAUDE_CODE_USE_BEDROCK", provider.adapter.command.inherited_environment)


if __name__ == "__main__":
    unittest.main()
