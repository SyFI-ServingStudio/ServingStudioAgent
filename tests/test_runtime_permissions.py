"""The Codex permission posture, and the ways it can silently stop being one."""

import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.providers.codex.home import CodexProfile
from vibesim_agent.runtime.invocation import InvocationHome, RoleContext
from vibesim_agent.runtime.permissions import (
    CONTAINER_PERMISSIONS,
    HOST_PROFILE,
    CodexPermissions,
    host_permissions,
    reject_retired_settings,
    validated,
)


class HostProfileTests(unittest.TestCase):
    def profile(self, **overrides):
        permissions = host_permissions(
            **{
                "git_dir": Path("/trees/base/.git/worktrees/wt-topic"),
                "git_common_dir": Path("/trees/base/.git"),
                "workspace_roots": (Path("/tmp/run"), Path("/home/u/.cargo")),
                "unix_sockets": ("/var/run/docker.sock",),
                **overrides,
            }
        )
        return permissions, tomllib.loads(permissions.config)["permissions"][HOST_PROFILE]

    def test_both_git_directories_are_granted_by_absolute_path(self):
        _, profile = self.profile()
        # `:workspace` leaves `.git` read-only, and in a worktree `.git` is a
        # file pointing outside the tree, so the documented `:workspace_roots`
        # override never reaches it: `git commit` fails on index.lock.
        self.assertEqual(
            profile["filesystem"],
            {
                "/trees/base/.git/worktrees/wt-topic": "write",
                "/trees/base/.git": "write",
            },
        )
        self.assertEqual(profile["extends"], ":workspace")

    def test_a_plain_checkout_does_not_emit_the_same_key_twice(self):
        # TOML rejects a duplicate key outright, and the whole profile then
        # fails to load -- in a non-worktree repository these are one directory.
        permissions, profile = self.profile(
            git_dir=Path("/tree/.git"), git_common_dir=Path("/tree/.git")
        )
        self.assertEqual(profile["filesystem"], {"/tree/.git": "write"})
        self.assertEqual(permissions.config.count("/tree/.git"), 1)

    def test_the_extra_roots_and_the_docker_socket_are_named_explicitly(self):
        _, profile = self.profile()
        self.assertEqual(
            profile["workspace_roots"], {"/tmp/run": True, "/home/u/.cargo": True}
        )
        # Equivalent to granting root, and deliberate: most profiled kernels
        # run through a containerized environment.
        self.assertEqual(
            profile["network"],
            {"enabled": True, "unix_sockets": {"/var/run/docker.sock": "allow"}},
        )

    def test_the_host_posture_answers_escalations_itself(self):
        permissions, _ = self.profile()
        self.assertEqual(permissions.profile, HOST_PROFILE)
        self.assertEqual(permissions.approval_policy, "on-request")
        self.assertEqual(permissions.reviewer, "auto_review")
        self.assertIs(validated(permissions), permissions)

    def test_the_container_posture_closes_escalation_instead(self):
        self.assertEqual(CONTAINER_PERMISSIONS.profile, ":danger-full-access")
        self.assertEqual(CONTAINER_PERMISSIONS.approval_policy, "never")
        self.assertIsNone(CONTAINER_PERMISSIONS.reviewer)
        self.assertEqual(CONTAINER_PERMISSIONS.config, "")

    def test_a_path_that_cannot_be_a_toml_key_is_refused(self):
        for directory in (Path(""), Path("/tree/.git\nevil")):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                self.profile(git_dir=directory)


class RetiredSettingTests(unittest.TestCase):
    """The failure that matters most: a sandbox switched off without saying so."""

    def test_either_retired_key_is_refused(self):
        for config in (
            'sandbox_mode = "workspace-write"\n',
            "[sandbox_workspace_write]\nnetwork_access = true\n",
            '  sandbox_mode="danger-full-access"',
        ):
            with self.subTest(config=config), self.assertRaisesRegex(
                ValueError, "silently disables"
            ):
                reject_retired_settings(config)

    def test_an_ordinary_configuration_passes(self):
        reject_retired_settings(
            'model_provider = "custom"\n[model_providers.custom]\nname = "Custom"\n'
        )


class ProfileDeliveryTests(unittest.TestCase):
    """The profile reaches Codex through the managed home, not the command line."""

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.source = self.root / "source"
        self.source.mkdir()
        self.home = InvocationHome(self.root / "role", "/role-home")
        self.permissions = host_permissions(
            git_dir=Path("/tree/.git"), git_common_dir=Path("/tree/.git")
        )

    def context(self, **overrides):
        return RoleContext(
            **{"skills": "/tree/skills", "codex_config": self.permissions.config, **overrides}
        )

    def test_the_profile_is_appended_to_the_user_configuration(self):
        (self.source / "config.toml").write_text('model_provider = "custom"\n')
        CodexProfile(self.source).prepare(self.home, self.context())
        parsed = tomllib.loads((self.home.host / "config.toml").read_text())
        self.assertEqual(parsed["model_provider"], "custom")
        self.assertIn(HOST_PROFILE, parsed["permissions"])

    def test_a_repeated_preparation_does_not_stack_the_profile(self):
        (self.source / "config.toml").write_text('model_provider = "custom"\n')
        profile = CodexProfile(self.source)
        for _ in range(3):
            profile.prepare(self.home, self.context())
        # The whole file is recopied each time, so the append has to start from
        # the source rather than from what the last turn left behind.
        written = (self.home.host / "config.toml").read_text()
        self.assertEqual(written.count(f"[permissions.{HOST_PROFILE}]"), 1)

    def test_a_container_turn_leaves_the_configuration_alone(self):
        (self.source / "config.toml").write_text('model_provider = "custom"\n')
        CodexProfile(self.source).prepare(self.home, RoleContext(skills="/workspace/skills"))
        self.assertEqual(
            (self.home.host / "config.toml").read_text(), 'model_provider = "custom"\n'
        )

    def test_a_user_configuration_that_disables_the_profile_stops_the_turn(self):
        # `PROFILE_ENTRIES` copies the user's whole config.toml, so a
        # `sandbox_mode` they set for themselves would take the profile with it.
        (self.source / "config.toml").write_text('sandbox_mode = "workspace-write"\n')
        with self.assertRaisesRegex(ValueError, "silently disables"):
            CodexProfile(self.source).prepare(self.home, self.context())

    def test_a_profile_configuration_must_name_its_own_profile(self):
        with self.assertRaisesRegex(ValueError, "must name its own profile"):
            validated(CodexPermissions(":workspace", "never", config="[permissions.x]"))


if __name__ == "__main__":
    unittest.main()
