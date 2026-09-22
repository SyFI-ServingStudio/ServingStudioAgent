import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.runtime.worktree import WorktreeError, WorktreeProvisioner


class WorktreeProvisionerTests(unittest.TestCase):
    """Runs real Git: the skip-worktree bit and the admin directory are the point.

    A recorded double can report that `update-index --skip-worktree` was called
    without the bit ever being set, and it cannot show that `rmtree` would strand
    `.git/worktrees/<name>`. The double is used only to inject failure.
    """

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.main = self.root / "ServingStudioSim"
        self.main.mkdir()
        self.environment = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.root),
            "XDG_CONFIG_HOME": str(self.root / "config"),
        }
        self.git(self.main, "init", "--template=")
        self.git(self.main, "checkout", "-b", "main")
        self.git(self.main, "config", "user.name", "Worktree test")
        self.git(self.main, "config", "user.email", "worktree@example.invalid")
        (self.main / "profiling").mkdir()
        (self.main / "profiling/profile.db").write_text("committed\n")
        (self.main / "trace").mkdir()
        (self.main / "trace/committed.csv").write_text("tracked\n")
        (self.main / ".gitignore").write_text("*.csv\n!trace/committed.csv\n")
        self.git(self.main, "add", "-A")
        self.git(self.main, "-c", "commit.gpgsign=false", "commit", "-m", "initial")
        # Working copy diverges from the commit exactly as a GPU run leaves it.
        (self.main / "profiling/profile.db").write_text("working copy\n")
        (self.main / "trace/aime_long.csv").write_text("untracked\n")

    def git(self, directory, *arguments, check=True):
        return subprocess.run(
            ["git", "-C", str(directory), "-c", "core.hooksPath=/dev/null", *arguments],
            env=self.environment,
            capture_output=True,
            text=True,
            check=check,
            timeout=30,
        ).stdout

    def provisioner(self, **overrides):
        return WorktreeProvisioner(
            self.main, process_environment=self.environment, **overrides
        )

    def admin_directory(self, name):
        return self.main / ".git/worktrees" / name

    def test_stages_working_copy_artifacts_and_protects_the_tracked_one(self):
        destination = self.root / "wt-topic"
        worktree = self.provisioner().create(destination, branch="wt-topic")

        self.assertEqual(worktree.path, destination)
        self.assertEqual(worktree.branch, "wt-topic")
        self.assertEqual(
            worktree.base_revision,
            self.git(self.main, "rev-parse", "HEAD").strip(),
        )
        # The richer working copy, not the committed snapshot.
        self.assertEqual(
            (destination / "profiling/profile.db").read_text(), "working copy\n"
        )
        # Ignored, not merely untracked -- `--exclude-standard` would drop it.
        self.assertEqual(
            (destination / "trace/aime_long.csv").read_text(), "untracked\n"
        )
        self.assertEqual((destination / "trace/committed.csv").read_text(), "tracked\n")

    def test_skip_worktree_bit_keeps_the_tree_clean(self):
        destination = self.root / "wt-topic"
        self.provisioner().create(destination, branch="wt-topic")

        listing = self.git(destination, "ls-files", "-v", "profiling/profile.db")
        self.assertTrue(listing.startswith("S "), listing)
        # Without the bit a 60+ MB modified binary would sit here, and
        # `git commit -am` would sweep up the primary tree's kernel cache.
        self.assertEqual(self.git(destination, "status", "--porcelain"), "")

    def test_ignored_environments_are_never_copied(self):
        (self.main / ".venv").mkdir()
        (self.main / ".venv/pyvenv.cfg").write_text("home = /elsewhere\n")
        (self.main / "target").mkdir()
        (self.main / "target/release").mkdir()
        destination = self.root / "wt-topic"
        self.provisioner().create(destination, branch="wt-topic")

        # A copied editable install would import Python from the other checkout.
        self.assertFalse((destination / ".venv").exists())
        self.assertFalse((destination / "target").exists())

    def test_rejects_names_and_destinations_that_git_cannot_hold(self):
        provisioner = self.provisioner()
        for branch in ("", "--force", "bad branch", "refs/heads/x/", "..", "a..b"):
            with self.subTest(branch=branch):
                with self.assertRaises(WorktreeError):
                    provisioner.create(self.root / f"wt-{len(branch)}", branch=branch)
        with self.assertRaisesRegex(WorktreeError, "nested"):
            provisioner.create(self.main / "wt-inside", branch="wt-inside")
        (self.root / "wt-taken").mkdir()
        with self.assertRaisesRegex(WorktreeError, "already exists"):
            provisioner.create(self.root / "wt-taken", branch="wt-taken")

    def test_duplicate_branch_fails_without_stranding_the_first_worktree(self):
        first = self.root / "wt-topic"
        self.provisioner().create(first, branch="wt-topic")
        with self.assertRaises(WorktreeError):
            self.provisioner().create(self.root / "wt-other", branch="wt-topic")
        self.assertTrue((first / "profiling/profile.db").is_file())

    def test_failed_staging_leaves_no_tree_branch_or_admin_directory(self):
        real = subprocess.run
        calls = []

        def failing(command, **options):
            calls.append(command)
            if command[0] == "cp":
                return subprocess.CompletedProcess(command, 1, "", "no space left")
            return real(command, **options)

        destination = self.root / "wt-topic"
        with self.assertRaisesRegex(WorktreeError, "stage a working-copy artifact"):
            self.provisioner(run=failing).create(destination, branch="wt-topic")

        self.assertFalse(destination.exists())
        self.assertNotIn("wt-topic", self.git(self.main, "branch", "--list"))
        # `rmtree` alone would strand this and burn the name permanently.
        self.assertFalse(self.admin_directory("wt-topic").exists())
        # The name is usable again, which is the whole point of the cleanup.
        self.provisioner().create(destination, branch="wt-topic")
        self.assertTrue((destination / "profiling/profile.db").is_file())

    def test_unprotectable_tracked_artifact_is_a_hard_failure(self):
        real = subprocess.run

        def failing(command, **options):
            if "update-index" in command:
                return subprocess.CompletedProcess(command, 1, "", "denied")
            return real(command, **options)

        with self.assertRaisesRegex(WorktreeError, "protect a staged tracked artifact"):
            self.provisioner(run=failing).create(
                self.root / "wt-topic", branch="wt-topic"
            )
        self.assertFalse((self.root / "wt-topic").exists())
        self.assertFalse(self.admin_directory("wt-topic").exists())

    def test_ambient_git_routing_cannot_redirect_provisioning(self):
        decoy = self.root / "decoy"
        decoy.mkdir()
        self.git(decoy, "init", "--template=")
        provisioner = WorktreeProvisioner(
            self.main,
            process_environment={
                **self.environment,
                "GIT_DIR": str(decoy / ".git"),
                "GIT_WORK_TREE": str(decoy),
            },
        )
        destination = self.root / "wt-topic"
        provisioner.create(destination, branch="wt-topic")

        self.assertIn("wt-topic", self.git(self.main, "branch", "--list"))
        self.assertNotIn("wt-topic", self.git(decoy, "branch", "--list"))

    def test_base_selects_the_commit_and_unknown_bases_are_refused(self):
        self.git(self.main, "checkout", "-b", "feature")
        (self.main / "later.txt").write_text("later\n")
        self.git(self.main, "add", "-A")
        self.git(self.main, "-c", "commit.gpgsign=false", "commit", "-m", "later")
        head = self.git(self.main, "rev-parse", "main").strip()

        worktree = self.provisioner().create(
            self.root / "wt-topic", branch="wt-topic", base="main"
        )
        self.assertEqual(worktree.base_revision, head)
        self.assertFalse((worktree.path / "later.txt").exists())
        with self.assertRaisesRegex(WorktreeError, "base revision"):
            self.provisioner().create(
                self.root / "wt-missing", branch="wt-missing", base="no-such-ref"
            )


if __name__ == "__main__":
    unittest.main()
