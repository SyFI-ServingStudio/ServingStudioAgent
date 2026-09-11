import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.runtime.workspace import WorkspaceSnapshot


class WorkspaceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.source = self.root / "source"
        self.source.mkdir()
        self.environment = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.root),
            "XDG_CONFIG_HOME": str(self.root / "config"),
        }
        self.git(self.source, "init", "--template=")
        self.git(self.source, "checkout", "-b", "source-branch")
        self.git(self.source, "config", "user.name", "Snapshot test")
        self.git(self.source, "config", "user.email", "snapshot@example.invalid")
        (self.source / "tracked").write_text("original\n")
        self.commit("initial")
        self.destination = self.root / "snapshot"

    def git(self, directory, *arguments, check=True):
        return subprocess.run(
            [
                "git",
                "-C",
                str(directory),
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                *arguments,
            ],
            env=self.environment,
            capture_output=True,
            text=True,
            check=check,
            timeout=10,
        ).stdout.strip()

    def commit(self, message):
        self.git(self.source, "add", "-A")
        self.git(self.source, "commit", "-m", message)

    def snapshot(self, **kwargs):
        return WorkspaceSnapshot(
            self.source, process_environment=self.environment, **kwargs
        )

    def source_state(self):
        return (
            self.git(self.source, "rev-parse", "HEAD"),
            self.git(self.source, "status", "--porcelain=v1", "--untracked-files=all"),
            (self.source / ".git/index").read_bytes(),
            (self.source / ".git/config").read_bytes(),
        )

    def test_dirty_tracked_deletion_untracked_modes_and_independent_main_commit(self):
        (self.source / "deleted").write_text("delete later")
        (self.source / "script").write_text("#!/bin/sh\nexit 0\n")
        self.commit("files")
        (self.source / "tracked").write_text("dirty working copy\n")
        (self.source / "deleted").unlink()
        (self.source / "script").chmod(0o755)
        (self.source / "untracked").write_text("not selected")
        before = self.source_state()
        result = self.snapshot().populate(self.destination)
        self.assertEqual(result, self.destination)
        self.assertEqual((result / "tracked").read_text(), "dirty working copy\n")
        self.assertFalse((result / "deleted").exists())
        self.assertFalse((result / "untracked").exists())
        self.assertEqual((result / "script").stat().st_mode & 0o777, 0o755)
        self.assertTrue(
            self.git(result, "ls-files", "--stage", "script").startswith("100755 ")
        )
        self.assertEqual(self.git(result, "branch", "--show-current"), "main")
        self.assertEqual(self.git(result, "rev-list", "--count", "HEAD"), "1")
        self.assertEqual(self.git(result, "status", "--porcelain"), "")
        self.assertEqual(self.git(result, "remote"), "")
        self.assertFalse((result / ".git/objects/info/alternates").exists())
        self.assertEqual(self.source_state(), before)
        (result / "tracked").write_text("independent")
        self.assertEqual((self.source / "tracked").read_text(), "dirty working copy\n")

    def test_gitlinks_are_reported_but_not_copied_or_initialized(self):
        commit = self.git(self.source, "rev-parse", "HEAD")
        self.git(
            self.source,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{commit},vendor/module",
        )
        module = self.source / "vendor/module"
        module.mkdir(parents=True)
        (module / "private-state").write_text("not copied")
        snapshot = self.snapshot()
        files, modules = snapshot.tracked_entries()
        self.assertEqual(modules, [Path("vendor/module")])
        self.assertIn(Path("tracked"), files)
        before = self.source_state()
        snapshot.populate(self.destination)
        self.assertFalse((self.destination / "vendor/module").exists())
        self.assertEqual((module / "private-state").read_text(), "not copied")
        self.assertEqual(self.source_state(), before)

    def test_internal_and_dangling_links_preserved_external_targets_materialized(self):
        external = self.root / "external"
        external.mkdir()
        (external / "file").write_text("external contents")
        directory = external / "directory"
        directory.mkdir()
        (directory / "nested").write_text("external tree")
        (directory / "nested-link").symlink_to("nested")
        (self.source / "internal-link").symlink_to("tracked")
        (self.source / "dangling-link").symlink_to("missing")
        (self.source / "outside-missing").symlink_to(external / "absent")
        (self.source / "outside-file").symlink_to(external / "file")
        (self.source / "outside-directory").symlink_to(
            directory, target_is_directory=True
        )
        self.commit("links")
        before = self.source_state()
        self.snapshot().populate(self.destination)
        for name in ("internal-link", "dangling-link", "outside-missing"):
            self.assertTrue((self.destination / name).is_symlink())
            self.assertEqual(
                os.readlink(self.destination / name), os.readlink(self.source / name)
            )
        self.assertFalse((self.destination / "outside-file").is_symlink())
        self.assertEqual(
            (self.destination / "outside-file").read_text(), "external contents"
        )
        self.assertFalse((self.destination / "outside-directory").is_symlink())
        self.assertEqual(
            (self.destination / "outside-directory/nested").read_text(), "external tree"
        )
        self.assertTrue(
            (self.destination / "outside-directory/nested-link").is_symlink()
        )
        self.assertEqual(self.source_state(), before)

    def test_existing_destination_and_source_nested_destination_never_overwritten(self):
        self.destination.mkdir()
        (self.destination / "keep").write_text("preserved")
        before = self.source_state()
        with self.assertRaises(FileExistsError):
            self.snapshot().populate(self.destination)
        self.assertEqual(list(self.destination.iterdir()), [self.destination / "keep"])
        self.assertEqual((self.destination / "keep").read_text(), "preserved")
        with self.assertRaises(ValueError):
            self.snapshot().populate(self.source / "nested")
        self.assertFalse((self.source / "nested").exists())
        self.assertEqual(self.source_state(), before)

    def test_unmerged_index_rejected_before_creating_destination(self):
        self.git(self.source, "checkout", "-b", "other")
        (self.source / "tracked").write_text("other\n")
        self.commit("other")
        self.git(self.source, "checkout", "source-branch")
        (self.source / "tracked").write_text("main\n")
        self.commit("main")
        self.git(self.source, "merge", "other", check=False)
        self.assertTrue(self.git(self.source, "ls-files", "--unmerged"))
        before = self.source_state()
        with self.assertRaisesRegex(ValueError, "unmerged"):
            self.snapshot().populate(self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.source_state(), before)

    def test_git_routing_environment_cannot_redirect_source_or_destination(self):
        other = self.root / "unrelated"
        other.mkdir()
        self.git(other, "init", "--template=")
        (other / "keep").write_text("untouched")
        environment = self.environment | {
            "GIT_DIR": str(other / ".git"),
            "GIT_WORK_TREE": str(other),
            "GIT_INDEX_FILE": str(self.root / "redirect-index"),
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.bare",
            "GIT_CONFIG_VALUE_0": "true",
        }
        before = self.source_state()
        snapshot = WorkspaceSnapshot(self.source, process_environment=environment)
        snapshot.populate(self.destination)
        self.assertEqual((self.destination / "tracked").read_text(), "original\n")
        self.assertEqual(self.git(self.destination, "branch", "--show-current"), "main")
        self.assertFalse((other / ".git/index").exists())
        self.assertFalse((self.root / "redirect-index").exists())
        self.assertEqual((other / "keep").read_text(), "untouched")
        self.assertEqual(self.source_state(), before)

    def test_global_worktree_and_bare_settings_cannot_redirect_snapshot_commit(self):
        other = self.root / "other-worktree"
        other.mkdir()
        (other / "private").write_text("must not enter snapshot")
        (self.source / "untracked").write_text("also excluded")
        before = self.source_state()
        config = self.root / ".gitconfig"
        for index, redirect in enumerate((self.source, other)):
            with self.subTest(redirect=redirect):
                config.write_text(f'[core]\nworktree = "{redirect}"\nbare = true\n')
                try:
                    destination = self.root / f"snapshot-{index}"
                    self.snapshot().populate(destination)
                finally:
                    config.unlink()
                self.assertEqual(
                    self.git(destination, "rev-parse", "--show-toplevel"),
                    str(destination),
                )
                self.assertEqual(
                    self.git(destination, "ls-tree", "--name-only", "HEAD"), "tracked"
                )
                self.assertEqual(self.git(destination, "status", "--porcelain"), "")
                self.assertEqual((destination / "tracked").read_text(), "original\n")
                self.assertEqual(self.source_state(), before)
                self.assertEqual(
                    (other / "private").read_text(), "must not enter snapshot"
                )

    def test_all_tracked_files_deleted_produces_empty_initial_commit(self):
        (self.source / "tracked").unlink()
        before = self.source_state()
        self.snapshot().populate(self.destination)
        self.assertEqual(
            self.git(self.destination, "ls-tree", "--name-only", "HEAD"), ""
        )
        self.assertEqual(self.git(self.destination, "rev-list", "--count", "HEAD"), "1")
        self.assertEqual(self.git(self.destination, "branch", "--show-current"), "main")
        self.assertEqual(self.source_state(), before)
