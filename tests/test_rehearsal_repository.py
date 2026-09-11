import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools import rehearsal_repository as rehearsal


class RehearsalRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.environment = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.root),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
        self.original = self.root / "original"
        self.original.mkdir()
        self.git(self.original, "init", "--template=", "--initial-branch=main")
        self.git(self.original, "config", "user.name", "Fixture")
        self.git(self.original, "config", "user.email", "fixture@example.invalid")
        (self.original / "tracked").write_text("committed\n")
        (self.original / "delete-me").write_text("delete later\n")
        self.git(self.original, "add", "-A")
        self.git(self.original, "commit", "-m", "initial")
        self.git(self.original, "tag", "keep-tag")
        self.source = self.root / "source"
        self.git(self.original, "worktree", "add", "-b", "work", str(self.source))
        self.target = self.root / "bundle"
        (self.source / "tracked").write_text("staged content\n")
        self.git(self.source, "add", "tracked")
        (self.source / "tracked").write_text("unstaged content\n")
        (self.source / "delete-me").unlink()
        (self.source / "untracked").write_bytes(b"untracked bytes\x00")
        (self.source / "build").mkdir()
        (self.source / "build/result").write_bytes(b"compiled artifact")

    def git(self, path, *args):
        result = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            env=self.environment,
            check=True,
            capture_output=True,
            timeout=15,
        )
        return result.stdout.decode().strip()

    def tree(self, path):
        return {
            str(entry.relative_to(path)): os.readlink(entry)
            if entry.is_symlink()
            else entry.read_bytes()
            for entry in path.rglob("*")
            if entry.is_symlink() or entry.is_file()
        }

    def prepare(self):
        with patch.dict(os.environ, self.environment, clear=True):
            return rehearsal.prepare_repository(
                self.source, self.target, source_quiesced=True
            )

    def test_worktree_copy_preserves_head_refs_index_and_dirty_untracked_bytes(self):
        source_before, metadata_before = (
            self.tree(self.source),
            self.tree(self.original / ".git"),
        )
        head = self.git(self.source, "rev-parse", "HEAD")
        index = self.git(self.source, "ls-files", "--stage")
        self.prepare()
        copied = self.target / "repo"
        self.assertTrue((copied / ".git").is_dir())
        self.assertEqual(self.git(copied, "rev-parse", "HEAD"), head)
        self.assertEqual(
            self.git(copied, "show-ref"), self.git(self.source, "show-ref")
        )
        self.assertEqual(self.git(copied, "ls-files", "--stage"), index)
        for path, content in source_before.items():
            if path != ".git":
                self.assertEqual((copied / path).read_bytes(), content)
        self.assertFalse((copied / "delete-me").exists())
        self.assertEqual(self.git(copied, "remote"), "")
        self.assertTrue((self.target / "evidence").is_dir())
        self.assertEqual(self.tree(self.source), source_before)
        self.assertEqual(self.tree(self.original / ".git"), metadata_before)
        for flag in ("--absolute-git-dir", "--git-common-dir"):
            located = Path(self.git(copied, "rev-parse", flag))
            if not located.is_absolute():
                located = copied / located
            self.assertTrue(located.resolve().is_relative_to(self.target))

    def test_target_add_commit_and_repack_do_not_modify_source_git_or_files(self):
        self.prepare()
        before = self.tree(self.source), self.tree(self.original / ".git")
        copied = self.target / "repo"
        source_inodes = {
            (entry.stat().st_dev, entry.stat().st_ino)
            for entry in (self.original / ".git").rglob("*")
            if entry.is_file()
        }
        target_inodes = {
            (entry.stat().st_dev, entry.stat().st_ino)
            for entry in (copied / ".git").rglob("*")
            if entry.is_file()
        }
        self.assertFalse(source_inodes & target_inodes)
        (copied / "target-only").write_text("independent")
        self.git(copied, "add", "-A")
        self.git(
            copied,
            "-c",
            "user.name=Target",
            "-c",
            "user.email=target@example.invalid",
            "commit",
            "-m",
            "target",
        )
        self.git(copied, "repack", "-a", "-d")
        self.assertEqual(
            (self.tree(self.source), self.tree(self.original / ".git")), before
        )
        self.assertNotEqual(
            self.git(copied, "rev-parse", "HEAD"),
            self.git(self.source, "rev-parse", "HEAD"),
        )

    def test_split_index_preserves_staged_and_intent_to_add_semantics(self):
        (self.source / "intent").write_text("not staged yet")
        self.git(self.source, "add", "-N", "intent")
        self.git(self.source, "update-index", "--split-index")
        before = self.tree(self.original / ".git")
        expected_index = self.git(self.source, "ls-files", "--stage", "--debug")
        expected_staged = self.git(self.source, "diff", "--cached", "--raw")
        self.prepare()
        copied = self.target / "repo"
        self.assertEqual(self.git(copied, "diff", "--cached", "--raw"), expected_staged)
        self.assertEqual(
            self.git(copied, "ls-files", "--stage"),
            self.git(self.source, "ls-files", "--stage"),
        )
        self.assertIn("intent", expected_index)
        self.assertEqual(
            self.git(copied, "diff", "--name-only"),
            self.git(self.source, "diff", "--name-only"),
        )
        self.assertEqual(self.tree(self.original / ".git"), before)

    def test_real_initialized_submodule_gets_independent_git_directory(self):
        sub = self.root / "sub-source"
        sub.mkdir()
        self.git(sub, "init", "--template=", "--initial-branch=main")
        (sub / "module-file").write_text("module committed")
        self.git(sub, "add", "-A")
        self.git(
            sub,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "sub",
        )
        self.git(
            self.source,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(sub),
            "vendor/module",
        )
        self.git(self.source, "commit", "-m", "submodule")
        module = self.source / "vendor/module"
        self.assertTrue((module / ".git").is_file())
        (module / "module-file").write_text("module dirty")
        (module / "untracked-module").write_text("module untracked")
        before = (
            self.tree(self.source),
            self.tree(self.original / ".git"),
            self.tree(sub / ".git"),
        )
        self.prepare()
        copied = self.target / "repo/vendor/module"
        self.assertTrue((copied / ".git").is_dir())
        self.assertEqual(
            self.git(copied, "rev-parse", "HEAD"), self.git(module, "rev-parse", "HEAD")
        )
        self.assertEqual((copied / "module-file").read_text(), "module dirty")
        self.assertEqual((copied / "untracked-module").read_text(), "module untracked")
        self.assertEqual(self.git(copied, "remote"), "")
        self.assertEqual(
            (
                self.tree(self.source),
                self.tree(self.original / ".git"),
                self.tree(sub / ".git"),
            ),
            before,
        )

    def test_existing_target_overlap_and_missing_quiescence_refuse_without_writes(self):
        before = self.tree(self.source), self.tree(self.original / ".git")
        with self.assertRaises(ValueError):
            rehearsal.prepare_repository(self.source, self.target)
        self.assertFalse(self.target.exists())
        for target in (
            self.source,
            self.source / "child",
            self.original / ".git/copied",
        ):
            with (
                self.subTest(target=target),
                self.assertRaises((ValueError, FileExistsError)),
            ):
                rehearsal.prepare_repository(self.source, target, source_quiesced=True)
            self.assertFalse((target / "repo").exists())
        self.target.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(list(self.target.iterdir()), [])
        self.assertEqual(
            (self.tree(self.source), self.tree(self.original / ".git")), before
        )

    def test_alternates_promisor_and_active_merge_are_rejected(self):
        gitdir = Path(self.git(self.source, "rev-parse", "--absolute-git-dir"))
        alternates = self.original / ".git/objects/info/alternates"
        alternates.write_text(str(self.original / ".git/objects") + "\n")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.target.exists())
        alternates.unlink()
        self.git(self.source, "config", "remote.origin.promisor", "true")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.target.exists())
        self.git(self.source, "config", "--unset", "remote.origin.promisor")
        (gitdir / "MERGE_HEAD").write_text(
            self.git(self.source, "rev-parse", "HEAD") + "\n"
        )
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.target.exists())

    def test_hostile_git_environment_cannot_redirect_repository_or_index(self):
        bogus = self.root / "bogus-index"
        bogus.write_bytes(b"must not touch")
        hostile = {
            **self.environment,
            "GIT_DIR": str(self.original / ".git"),
            "GIT_WORK_TREE": str(self.original),
            "GIT_INDEX_FILE": str(bogus),
        }
        before = self.tree(self.original)
        with patch.dict(os.environ, hostile, clear=True):
            rehearsal.prepare_repository(self.source, self.target, source_quiesced=True)
        self.assertEqual(bogus.read_bytes(), b"must not touch")
        self.assertEqual(self.tree(self.original), before)
        self.assertEqual(
            (self.target / "repo/tracked").read_text(), "unstaged content\n"
        )

    def test_source_without_head_and_dangling_target_are_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        self.git(empty, "init", "--template=")
        with self.assertRaises(ValueError):
            rehearsal.prepare_repository(empty, self.target, source_quiesced=True)
        self.assertFalse(self.target.exists())
        missing = self.root / "not-created"
        self.target.symlink_to(missing, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertTrue(self.target.is_symlink())
        self.assertFalse(missing.exists())

    def test_local_filters_external_diff_and_fsmonitor_never_execute(self):
        marker = self.root / "executed"
        script = self.root / "malicious-hook"
        script.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 0\n")
        script.chmod(0o755)
        for key in (
            "core.fsmonitor",
            "diff.external",
            "diff.bad.command",
            "diff.bad.textconv",
            "filter.bad.clean",
        ):
            self.git(self.source, "config", key, str(script))
        (self.source / ".gitattributes").write_text("tracked filter=bad diff=bad\n")
        before = self.tree(self.source), self.tree(self.original / ".git")
        with self.assertRaisesRegex(
            rehearsal.RepositoryPreparationError,
            "configured filters or promisor remotes",
        ):
            self.prepare()
        self.assertFalse(self.target.exists())
        self.assertFalse(marker.exists())
        self.assertEqual(
            (self.tree(self.source), self.tree(self.original / ".git")), before
        )

    def test_core_configuration_and_local_rules_preserve_git_interpretation(self):
        self.git(self.source, "config", "core.filemode", "false")
        self.git(self.source, "config", "core.autocrlf", "true")
        (self.source / "tracked").chmod(0o755)
        normalized = self.source / "normalized"
        normalized.write_bytes(b"first\nsecond\n")
        self.git(self.source, "add", "normalized")
        normalized.write_bytes(b"first\r\nsecond\r\n")
        rules = {}
        for name, content in (
            ("exclude", b"# local rule\nlocal-ignored\n"),
            ("attributes", b"tracked -text\n"),
        ):
            path = Path(
                self.git(
                    self.source,
                    "rev-parse",
                    "--path-format=absolute",
                    "--git-path",
                    "info/" + name,
                )
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            rules[name] = content
        (self.source / "local-ignored").write_bytes(b"preserved ignored bytes")
        before = self.tree(self.source), self.tree(self.original / ".git")
        self.prepare()
        copied = self.target / "repo"
        for repo in (self.source, copied):
            self.assertEqual(
                self.git(repo, "config", "--bool", "core.filemode"), "false"
            )
            self.assertEqual(self.git(repo, "config", "core.autocrlf"), "true")
            self.assertEqual(
                self.git(repo, "diff", "--name-only", "--", "normalized"), ""
            )
            self.assertEqual(self.git(repo, "diff", "--summary", "--", "tracked"), "")
            self.assertEqual(
                self.git(repo, "check-ignore", "local-ignored"), "local-ignored"
            )
            self.assertEqual(
                self.git(repo, "check-attr", "text", "--", "tracked"),
                "tracked: text: unset",
            )
        self.assertEqual((copied / "normalized").read_bytes(), normalized.read_bytes())
        self.assertEqual(
            (copied / "local-ignored").read_bytes(), b"preserved ignored bytes"
        )
        for name, content in rules.items():
            self.assertEqual((copied / ".git/info" / name).read_bytes(), content)
        self.assertEqual(
            (self.tree(self.source), self.tree(self.original / ".git")), before
        )

    def test_explicit_empty_filemode_keeps_git_false_semantics(self):
        self.git(self.source, "config", "core.filemode", "")
        (self.source / "tracked").chmod(0o755)
        self.assertEqual(
            self.git(self.source, "config", "--bool", "core.filemode"), "false"
        )
        self.assertEqual(
            self.git(self.source, "diff", "--summary", "--", "tracked"), ""
        )
        self.prepare()
        copied = self.target / "repo"
        self.assertEqual(self.git(copied, "config", "--bool", "core.filemode"), "false")
        self.assertEqual(self.git(copied, "diff", "--summary", "--", "tracked"), "")

    def test_valueless_sparse_checkout_true_requires_explicit_preparation(self):
        config = self.original / ".git/config"
        config.write_bytes(config.read_bytes() + b"\n[core]\n sparseCheckout\n")
        self.assertEqual(
            self.git(self.source, "config", "--bool", "core.sparseCheckout"), "true"
        )
        before = self.tree(self.source), self.tree(self.original / ".git")
        with self.assertRaisesRegex(
            rehearsal.RepositoryPreparationError, "sparse repositories"
        ):
            self.prepare()
        self.assertFalse(self.target.exists())
        self.assertEqual(
            (self.tree(self.source), self.tree(self.original / ".git")), before
        )

    def test_external_git_rules_require_explicit_preparation(self):
        external = self.root / "external-rules"
        external.write_bytes(b"external rule\n")
        for key in ("core.excludesFile", "core.attributesFile"):
            with self.subTest(key=key):
                self.git(self.source, "config", key, str(external))
                before = self.tree(self.source), self.tree(self.original / ".git")
                with self.assertRaisesRegex(
                    rehearsal.RepositoryPreparationError, "external Git rule files"
                ):
                    self.prepare()
                self.assertFalse(self.target.exists())
                self.assertEqual(
                    (self.tree(self.source), self.tree(self.original / ".git")), before
                )
                self.assertEqual(external.read_bytes(), b"external rule\n")
                self.git(self.source, "config", "--unset", key)

    def test_global_core_worktree_cannot_retarget_copy(self):
        global_config = self.root / ".gitconfig"
        global_config.write_text(f"[core]\n worktree = {self.original}\n")
        environment = dict(self.environment)
        environment.pop("GIT_CONFIG_GLOBAL")
        before = self.tree(self.original)
        with patch.dict(os.environ, environment, clear=True):
            rehearsal.prepare_repository(self.source, self.target, source_quiesced=True)
        self.assertEqual(
            (self.target / "repo/tracked").read_text(), "unstaged content\n"
        )
        self.assertEqual(self.tree(self.original), before)

    def test_uninitialized_gitlink_contents_are_preserved_and_reported(self):
        head = self.git(self.source, "rev-parse", "HEAD")
        self.git(
            self.source,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{head},vendor/uninitialized",
        )
        module = self.source / "vendor/uninitialized"
        module.mkdir(parents=True)
        (module / "preserved").write_text("uninitialized working files")
        report = self.prepare()
        self.assertTrue(report["git_verified"])
        self.assertFalse(report["execution_isolation_verified"])
        self.assertEqual(report["uninitialized_gitlinks"], ["vendor/uninitialized"])
        self.assertEqual(
            (self.target / "repo/vendor/uninitialized/preserved").read_text(),
            "uninitialized working files",
        )
        self.assertFalse((self.target / "repo/vendor/uninitialized/.git").exists())

    def test_final_git_verification_failure_removes_owned_bundle_after_clone(self):
        verify = rehearsal._verify_git
        final_checks = []
        before = self.tree(self.source), self.tree(self.original / ".git")
        children = set(self.root.iterdir())

        def fail_final(repo, snapshot):
            if repo == self.target / "repo":
                final_checks.append(repo)
                self.assertTrue((repo / ".git/objects").is_dir())
                raise rehearsal.RepositoryPreparationError(
                    "injected final verification failure"
                )
            return verify(repo, snapshot)

        with (
            patch.object(rehearsal, "_verify_git", side_effect=fail_final),
            self.assertRaisesRegex(
                rehearsal.RepositoryPreparationError,
                "injected final verification failure",
            ),
        ):
            self.prepare()
        self.assertEqual(final_checks, [self.target / "repo"])
        self.assertFalse(self.target.exists())
        self.assertEqual(set(self.root.iterdir()), children)
        self.assertEqual(
            (self.tree(self.source), self.tree(self.original / ".git")), before
        )
