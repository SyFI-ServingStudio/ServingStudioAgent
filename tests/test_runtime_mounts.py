"""Bind mount composition preserves permissions without mutating shared paths."""

import unittest
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vibesim_agent.runtime.mounts import HF_TARGET, Mount, workspace_mounts
from vibesim_agent.settings import ContainerSettings


class RuntimeMountTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        for name in ("workspace", "main", "prompts", "mcp", "peer", "hf"):
            (self.root / name).mkdir()
        self.workspace = self.root / "workspace"
        (self.workspace / "AGENTS.md").write_text("host instructions")
        self.prompt = self.root / "prompts" / "AGENTS.single.md"
        self.prompt.write_text("selected instructions")
        self.submodule = self.root / "main" / "external" / "library"
        self.submodule.mkdir(parents=True)
        self.settings = ContainerSettings(
            image="runner", uid=1000, gid=1000, user="runner", home=Path("/home/runner")
        )

    def mounts(self, **overrides):
        options = {
            "workspace": self.workspace,
            "main": self.root / "main",
            "submodules": [Path("external/library")],
            "prompts": self.root / "prompts",
            "agent_prompt": self.prompt,
            "mcp": self.root / "mcp",
            "settings": self.settings,
        }
        options.update(overrides)
        return workspace_mounts(**options)

    def snapshot(self):
        return {
            str(path.relative_to(self.root)): (
                path.stat().st_mtime_ns,
                path.read_bytes() if path.is_file() else None,
            )
            for path in self.root.rglob("*")
        }

    def test_permissions_and_destinations_match_legacy_without_host_writes(self):
        before = self.snapshot()
        mounts = self.mounts(
            peer_workspace=self.root / "peer",
            settings=self.settings.model_copy(update={"hf_home": self.root / "hf"}),
        )
        self.assertEqual(
            [(str(m.target), m.read_only) for m in mounts],
            [
                ("/workspace", False),
                ("/opt/vibesim/analyzer-evidence-mcp", True),
                ("/opt/vibesim/prompts", True),
                ("/workspace/AGENTS.md", True),
                ("/workspace/external/library", True),
                ("/candidate", True),
                ("/model", True),
            ],
        )
        self.assertEqual(mounts[3].source, self.prompt)
        self.assertEqual(mounts[4].source, self.submodule)
        for mount in mounts:
            self.assertEqual(mount.docker_args()[0], "--mount")
            self.assertEqual("readonly" in mount.docker_args()[1], mount.read_only)
        self.assertEqual(self.snapshot(), before)

    def test_missing_peer_or_file_skips_but_configured_missing_hf_rejects(self):
        baseline = self.mounts()
        for peer in (self.root / "missing", self.prompt):
            self.assertEqual(self.mounts(peer_workspace=peer), baseline)
        for cache in (self.root / "missing", self.prompt):
            with self.assertRaisesRegex(ValueError, "HF_HOME"):
                self.mounts(
                    settings=self.settings.model_copy(update={"hf_home": cache})
                )
        self.assertFalse((self.root / "missing").exists())
        self.assertNotIn(HF_TARGET, {mount.target for mount in baseline})

    def test_agents_target_missing_directory_or_symlink_fails_without_mutation(self):
        target = self.workspace / "AGENTS.md"
        target.unlink()
        for state in ("missing", "directory", "symlink"):
            with self.subTest(state=state):
                if state == "directory":
                    target.mkdir()
                elif state == "symlink":
                    target.rmdir()
                    target.symlink_to(self.prompt)
                with self.assertRaisesRegex(
                    ValueError, "refusing to modify shared workspace"
                ):
                    self.mounts()
                if state == "missing":
                    self.assertFalse(target.exists())
        self.assertEqual(self.prompt.read_text(), "selected instructions")

    def test_required_mount_sources_reject_missing_and_relative_paths(self):
        for key in ("workspace", "main", "prompts", "agent_prompt", "mcp"):
            for path in (self.root / "missing", Path("relative")):
                with self.subTest(key=key, path=path), self.assertRaises(ValueError):
                    self.mounts(**{key: path})

    def test_mount_arguments_keep_spaces_colons_and_equals_in_one_field(self):
        source = self.root / "path with spaces:equals=here"
        source.mkdir()
        mount = Mount(source, PurePosixPath("/target with spaces:here"), read_only=True)
        self.assertEqual(
            mount.docker_args(),
            [
                "--mount",
                f"type=bind,src={source},dst=/target with spaces:here,readonly",
            ],
        )

    def test_mount_csv_injection_and_control_characters_are_rejected(self):
        for value in (
            "bad,readonly",
            'bad"quote',
            "bad\nline",
            "bad\rline",
            "bad\tline",
            "bad\x00value",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "mount paths"):
                    Mount(self.root / value, PurePosixPath("/target"))
                with self.assertRaisesRegex(ValueError, "mount paths"):
                    Mount(self.root, PurePosixPath("/" + value))
        source = self.root / "bad,source"
        source.mkdir()
        alias = self.root / "safe-alias"
        alias.symlink_to(source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "mount paths"):
            Mount(alias, PurePosixPath("/target"))

    def test_submodule_absolute_traversal_duplicate_and_symlink_escape_rejected(self):
        for paths in (
            [Path("/external")],
            [Path("../peer")],
            [Path(".")],
            [Path("AGENTS.md")],
            [Path("external/library")] * 2,
        ):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                self.mounts(submodules=paths)
        (self.root / "main" / "escape").symlink_to(
            self.root / "peer", target_is_directory=True
        )
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.mounts(submodules=[Path("escape")])

    def test_mount_sources_are_canonical_and_targets_cannot_escape(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root / "hf", target_is_directory=True)
        self.assertEqual(Mount(alias, PurePosixPath("/model")).source, self.root / "hf")
        for target in ("/", "//", "relative", "/workspace/../etc"):
            with self.assertRaises(ValueError):
                Mount(self.root, PurePosixPath(target))
        with self.assertRaises(ValueError):
            Mount(self.root / "missing", PurePosixPath("/target"))

    def test_mount_composition_does_not_read_environment_or_launch_processes(self):
        with (
            patch("os.getenv", side_effect=AssertionError("environment read")),
            patch("os.environ", {}),
            patch("subprocess.run", side_effect=AssertionError("process launch")),
        ):
            self.assertTrue(self.mounts())
