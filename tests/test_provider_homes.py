import json
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.runtime_fixtures import agent_request, execution_environment
from vibesim_agent.domain.roles import Role
from vibesim_agent.providers.claude.home import ClaudeProfile
from vibesim_agent.providers.codex.command import CodexCommand
from vibesim_agent.providers.codex.home import CodexProfile
from vibesim_agent.runtime.homes import role_home
from vibesim_agent.runtime.invocation import InvocationHome


class ProviderHomesTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.source = self.root / "source"
        self.source.mkdir()
        self.home = InvocationHome(self.root / "role", "/role home")

    def test_codex_refreshes_only_profile_and_preserves_runtime_state(self):
        (self.source / "auth.json").write_text('{"token":"old"}')
        (self.source / "sessions").mkdir()
        (self.source / "sessions/host.json").write_text("host conversation")
        (self.source / "history.jsonl").write_text("host history")
        profile = CodexProfile(self.source)
        profile.prepare(self.home)
        self.assertFalse((self.home.host / "sessions/host.json").exists())
        self.assertFalse((self.home.host / "history.jsonl").exists())
        saved = self.home.host / "sessions/saved.json"
        saved.write_text("role conversation")
        marker = self.home.host / "call-cancelled.pid.cancel"
        marker.touch()
        (self.source / "auth.json").write_text('{"token":"new"}')
        profile.prepare(self.home)
        self.assertEqual(
            json.loads((self.home.host / "auth.json").read_text())["token"], "new"
        )
        self.assertEqual(saved.read_text(), "role conversation")
        self.assertTrue(marker.exists())
        (self.source / "auth.json").unlink()
        profile.prepare(self.home)
        self.assertFalse((self.home.host / "auth.json").exists())

    def test_codex_catalog_override_preserves_original_toml(self):
        config = '"model_catalog_json" = "/host/catalog.json" # comment\n[model_providers.custom]\nname="Custom"\n'
        (self.source / "config.toml").write_text(config)
        (self.source / "models_catalog.json").write_text("{}")
        profile = CodexProfile(self.source)
        profile.prepare(self.home)
        builder = replace(
            CodexCommand(execution_environment()),
            catalog_filename=profile.catalog_filename,
        )
        command = builder.build(
            agent_request(), home=self.home.container
        )
        settings = [command[i + 1] for i, part in enumerate(command) if part == "-c"]
        parsed = tomllib.loads("\n".join(settings))
        self.assertEqual(parsed["model_catalog_json"], "/role home/models_catalog.json")
        self.assertEqual((self.home.host / "config.toml").read_text(), config)

    def test_separate_roles_do_not_share_sessions(self):
        profile = CodexProfile(self.source)
        profile.prepare(self.home)
        saved = self.home.host / "sessions/saved.json"
        saved.write_text("first")
        second = InvocationHome(self.root / "second", "/second")
        profile.prepare(second)
        self.assertEqual(list((second.host / "sessions").iterdir()), [])
        self.assertEqual(saved.read_text(), "first")

    def test_stale_rules_removed_but_cache_and_sessions_retained(self):
        rules = self.source / "rules"
        rules.mkdir()
        (rules / "default.rules").write_text("old rules")
        profile = CodexProfile(self.source)
        profile.prepare(self.home)
        (self.home.host / "cache/durable").write_text("cache")
        (rules / "default.rules").unlink()
        rules.rmdir()
        profile.prepare(self.home)
        self.assertFalse((self.home.host / "rules").exists())
        self.assertEqual((self.home.host / "cache/durable").read_text(), "cache")

    def test_runtime_directory_symlink_is_not_followed(self):
        self.home.host.mkdir()
        external = self.root / "external"
        external.mkdir()
        (external / "saved").write_text("unchanged")
        (self.home.host / "sessions").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not be symlinks"):
            CodexProfile(self.source).prepare(self.home)
        self.assertEqual(list(external.iterdir()), [external / "saved"])
        self.assertEqual((external / "saved").read_text(), "unchanged")

    def test_scope_and_role_determine_the_same_host_and_container_location(self):
        root = self.root / "conversation"
        one = role_home(root, "/runtime", Role.ASSISTANT, "codex:https://provider/v1")
        self.assertEqual(
            one,
            role_home(root, "/runtime", Role.ASSISTANT, "codex:https://provider/v1"),
        )
        self.assertNotEqual(
            one, role_home(root, "/runtime", Role.ASSISTANT, "codex:https://other/v1")
        )
        self.assertNotEqual(
            one,
            role_home(root, "/runtime", Role.IMPLEMENTER, "codex:https://provider/v1"),
        )
        relative = one.host.relative_to(root)
        self.assertEqual(one.container, "/runtime/" + relative.as_posix())
        hostile = role_home(root, "/runtime", Role.ASSISTANT, "../../untrusted")
        self.assertTrue(hostile.host.is_relative_to(root))
        self.assertFalse(root.exists())
        with self.assertRaises(ValueError):
            role_home(root, "/runtime", Role.ASSISTANT, "")

    def test_source_overlap_and_runtime_symlink_are_rejected(self):
        profile = CodexProfile(self.source)
        for target in (self.source, self.source / "nested", self.root):
            with self.subTest(target=target), self.assertRaises(ValueError):
                profile.prepare(InvocationHome(target, "/home"))
        self.home.host.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            profile.prepare(self.home)
        with self.assertRaises(ValueError):
            ClaudeProfile().prepare(self.home)

    def test_scoped_role_paths_reject_symlinks_before_profile_preparation(self):
        root = self.root / "conversation"
        root.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        mapped = role_home(root, "/runtime", Role.ASSISTANT, "scope")
        role = root / "assistant"
        role.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not be symlinks"):
            role_home(root, "/runtime", Role.ASSISTANT, "scope")
        role.unlink()
        role.mkdir()
        mapped.host.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not be symlinks"):
            role_home(root, "/runtime", Role.ASSISTANT, "scope")
        self.assertEqual(list(outside.iterdir()), [])

    def test_claude_preserves_existing_sessions_and_creates_container_skills_link(self):
        profile = ClaudeProfile()
        profile.prepare(self.home)
        saved = self.home.host / "sessions.json"
        saved.write_text("saved")
        profile.prepare(self.home)
        self.assertEqual(saved.read_text(), "saved")
        self.assertEqual(
            (self.home.host / "skills").readlink(), Path("/workspace/skills")
        )
        self.assertEqual(self.home.host.stat().st_mode & 0o777, 0o700)
        self.assertFalse((self.home.host / "auth.json").exists())
