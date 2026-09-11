"""Run the real smoke shell and Git copy with a fake Docker boundary."""

import json
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.bootstrap import configuration


class RunnerImageSmokeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.repo = Path(__file__).resolve().parents[1]
        self.main = self.root / "source with spaces"
        self.main.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "docker.jsonl"
        self.env = {
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "HOME": str(self.root / "private-home"),
            "TMPDIR": str(self.root),
            "VIBESIM_AGENT_MAIN_DIR": str(self.main),
            "SMOKE_TEST_LOG": str(self.log),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        self.git("init", "--template=", "--initial-branch=main")
        (self.main / "tracked").write_text("initial\n")
        (self.main / "removed").write_text("remove\n")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=f@invalid",
            "commit",
            "-qm",
            "fixture",
        )
        (self.main / "tracked").write_text("dirty bytes\n")
        (self.main / "removed").unlink()
        (self.main / "untracked").write_text("excluded\n")
        self.write_executable(
            "uv", "#!/bin/sh\nshift 3\nexec " + shlex.quote(sys.executable) + ' "$@"\n'
        )
        self.write_executable(
            "docker",
            "#!"
            + sys.executable
            + "\n"
            + """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
row = {'args': args}
if args[0] == 'run':
    copied = Path(args[args.index('--volume') + 1].removesuffix(':/workspace'))
    if os.environ.get('SMOKE_TEST_DIAGNOSTICS'):
        (copied/'logs').mkdir()
        (copied/'logs'/'failure.log').write_text('predictor error')
    row.update(copy=str(copied), content=(copied/'tracked').read_text(),
               untracked=(copied/'untracked').exists(), removed=(copied/'removed').exists(),
               git=(copied/'.git').is_dir(), script=sys.stdin.read())
with open(os.environ['SMOKE_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps(row)+'\\n')
raise SystemExit(17 if os.environ.get('SMOKE_TEST_FAIL') == args[0] else 0)
""",
        )

    def write_executable(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o700)

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.main), *args],
            env=self.env,
            check=True,
            capture_output=True,
        )

    def snapshot(self):
        return {
            str(p.relative_to(self.main)): p.read_bytes()
            for p in self.main.rglob("*")
            if p.is_file()
        }

    def invoke(self, *args):
        return subprocess.run(
            ["bash", str(self.repo / "scripts/test-runner-image.sh"), *args],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def calls(self):
        return (
            [json.loads(line) for line in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )

    def test_defaults_without_user_share_settings_and_copy_dirty_tracked_only(self):
        expected = configuration(environment=self.env, repo_root=self.repo).container
        before = self.snapshot()
        self.env["GIT_DIR"] = str(self.root / "foreign-git")
        self.env["GIT_INDEX_FILE"] = str(self.root / "foreign-index")
        result = self.invoke("--main-dir", str(self.main))
        self.assertEqual(result.returncode, 0, result.stderr)
        inspect, run = self.calls()
        self.assertEqual(inspect["args"], ["image", "inspect", expected.image])
        args = run["args"]
        self.assertEqual(
            args[args.index("--user") + 1], f"{expected.uid}:{expected.gid}"
        )
        self.assertIn(f"HOME={expected.home}", args)
        self.assertIn(f"USER={expected.user}", args)
        self.assertNotIn("--gpus", args)
        self.assertIn("-i", args)
        self.assertEqual(run["content"], "dirty bytes\n")
        self.assertFalse(run["untracked"])
        self.assertFalse(run["removed"])
        self.assertTrue(run["git"])
        self.assertIn(
            "Cargo target seed caused a third-party dependency", run["script"]
        )
        self.assertIn("cargo cc claude codex", run["script"])
        self.assertFalse(Path(run["copy"]).exists())
        self.assertEqual(before, self.snapshot())

    def test_timing_uses_explicit_new_identity_and_gpu_selection(self):
        self.env.update(
            VIBESIM_RUNNER_IMAGE="runner:test",
            VIBESIM_RUNNER_USER="smokeuser",
            VIBESIM_RUNNER_UID="1234",
            VIBESIM_RUNNER_GID="2345",
            VIBESIM_RUNNER_GPUS="device=2",
        )
        result = self.invoke("timing")
        self.assertEqual(result.returncode, 0, result.stderr)
        run = self.calls()[-1]
        self.assertIn("1234:2345", run["args"])
        self.assertIn("HOME=/home/smokeuser", run["args"])
        self.assertEqual(run["args"][run["args"].index("--gpus") + 1], "device=2")
        self.assertEqual(run["args"][-1], "timing")
        self.assertIn(
            str(self.repo / "scripts/lib/runner-timing-smoke.py")
            + ":/opt/vibesim/runner-timing-smoke.py:ro",
            run["args"],
        )
        self.assertIn("uv run python /opt/vibesim/runner-timing-smoke.py", run["script"])

    def test_empty_gpu_timing_and_invalid_configuration_fail_before_docker(self):
        for update, args, error in (
            (
                {"VIBESIM_RUNNER_GPUS": ""},
                ["timing"],
                "requires nonempty VIBESIM_RUNNER_GPUS",
            ),
            ({"VIBESIM_RUNNER_UID": "bad-secret"}, [], "VIBESIM_RUNNER_UID"),
            ({"CODEX_DOCKER_IMAGE": "bad-secret"}, [], "Retired Agent"),
        ):
            with self.subTest(update=tuple(update)):
                old = self.env.copy()
                self.env.update(update)
                result = self.invoke(*args)
                self.env = old
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertNotIn("bad-secret", result.stderr)
                self.assertEqual(self.calls(), [])
                self.assertEqual(list(self.root.glob("vibesim-runner-smoke.*")), [])

    def test_docker_and_copy_failures_do_not_report_success_or_leave_private_copy(self):
        before = self.snapshot()
        for failure in ("image", "run", "copy"):
            with self.subTest(failure=failure):
                self.env["SMOKE_TEST_FAIL"] = failure
                if failure == "copy":
                    self.env["VIBESIM_AGENT_MAIN_DIR"] = str(self.root / "missing")
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("runner image smoke passed", result.stdout)
                self.assertEqual(list(self.root.glob("vibesim-runner-smoke.*")), [])
                self.assertEqual(before, self.snapshot())

    def test_failed_run_preserves_diagnostics_and_exit_status(self):
        self.env.update(SMOKE_TEST_FAIL="run", SMOKE_TEST_DIAGNOSTICS="1")
        before = self.snapshot()
        result = self.invoke()
        self.assertEqual(result.returncode, 17)
        copied = Path(self.calls()[-1]["copy"])
        self.assertEqual((copied / "logs/failure.log").read_text(), "predictor error")
        self.assertIn(str(copied.parent), result.stderr)
        self.assertNotIn("runner image smoke passed", result.stdout)
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
