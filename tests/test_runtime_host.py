"""Host PATH readiness: a hard failure for absence, a warning for drift."""

import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.runtime.host import (
    PGID_SUFFIX,
    PINNED_VERSIONS,
    HostUnavailable,
    check_host_binaries,
    reap_process_groups,
    record_process_group,
)


class Recorder:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, command, **options):
        self.calls.append(list(command))
        outcome = self.results.pop(0) if self.results else ""
        if isinstance(outcome, Exception):
            raise outcome
        return subprocess.CompletedProcess(command, 0, stdout=outcome, stderr="")


class HostReadinessTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.logger = logging.getLogger("host-readiness-test")

    def tool(self, name, *, content=b""):
        # A real file, because the version cache is keyed by size and mtime.
        path = self.root / name
        path.write_bytes(content)
        return path

    def which(self, **available):
        return lambda name: available.get(name)

    def test_a_missing_tool_refuses_the_turn_before_anything_is_spawned(self):
        run = Recorder()
        with self.assertRaisesRegex(HostUnavailable, "missing required tool: codex"):
            check_host_binaries(
                ("codex",), logger=self.logger, which=self.which(), run=run
            )
        self.assertEqual(run.calls, [])

    def test_a_matching_version_is_reported_without_a_warning(self):
        path = str(self.tool("codex"))
        run = Recorder(f"codex-cli {PINNED_VERSIONS['codex']}\n")
        with self.assertNoLogs(self.logger, level="WARNING"):
            resolved = check_host_binaries(
                ("codex",),
                logger=self.logger,
                which=self.which(codex=path),
                run=run,
            )
        self.assertEqual(resolved, {"codex": path})
        self.assertEqual(run.calls, [[path, "--version"]])

    def test_a_version_that_differs_from_the_pin_warns_and_still_proceeds(self):
        path = str(self.tool("codex", content=b"different"))
        run = Recorder("codex-cli 0.144.0\n")
        with self.assertLogs(self.logger, level="WARNING") as logs:
            check_host_binaries(
                ("codex",),
                logger=self.logger,
                which=self.which(codex=path),
                run=run,
            )
        [message] = logs.output
        self.assertIn("0.144.0", message)
        self.assertIn(PINNED_VERSIONS["codex"], message)

    def test_an_unreadable_version_warns_rather_than_failing(self):
        path = str(self.tool("claude", content=b"unreadable"))
        run = Recorder(OSError("no such file"))
        with self.assertLogs(self.logger, level="WARNING"):
            check_host_binaries(
                ("claude",),
                logger=self.logger,
                which=self.which(claude=path),
                run=run,
            )

    def test_an_unpinned_tool_is_resolved_without_being_run(self):
        path = str(self.tool("git"))
        run = Recorder()
        check_host_binaries(
            ("git",), logger=self.logger, which=self.which(git=path), run=run
        )
        self.assertEqual(run.calls, [])

    def test_the_same_binary_is_only_interrogated_once(self):
        path = str(self.tool("codex", content=b"cached"))
        run = Recorder(f"codex-cli {PINNED_VERSIONS['codex']}\n")
        which = self.which(codex=path)
        for _ in range(3):
            check_host_binaries(("codex",), logger=self.logger, which=which, run=run)
        # Two node startups per turn is a cost the container mode does not pay,
        # and the answer only changes when the file does.
        self.assertEqual(len(run.calls), 1)


class OrphanReapingTests(unittest.TestCase):
    """What a host turn leaves behind when the backend dies mid-turn."""

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.logger = logging.getLogger("orphan-reaping-test")

    def group(self):
        """A leader with a child, the shape an interrupted CLI turn leaves."""
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
                "print(child.pid, flush=True);"
                "time.sleep(30)",
            ],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.addCleanup(self._terminate, process)
        return process, int(process.stdout.readline().strip())

    def _terminate(self, process):
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=5)
        process.stdout.close()

    def record(self, process, name="call-e1"):
        path = self.root / (name + PGID_SUFFIX)
        record_process_group(path, process.pid)
        return path

    def test_a_whole_group_is_reaped_not_just_its_leader(self):
        process, child = self.group()
        path = self.record(process)
        self.assertEqual(path.read_text().split()[0], str(os.getpgid(process.pid)))

        self.assertEqual(reap_process_groups(self.root, logger=self.logger), 1)
        process.wait(timeout=5)
        # The child is the one still writing into the worktree.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        with self.assertRaises(ProcessLookupError):
            os.kill(child, 0)
        # Consumed, so a second recovery cannot act on a stale number.
        self.assertFalse(path.exists())

    def test_a_reused_process_id_is_left_alone(self):
        process, _ = self.group()
        path = self.record(process)
        pgid = path.read_text().split()[0]
        # Same number, different process: exactly what PID reuse looks like.
        path.write_text(f"{pgid} 1\n")
        with self.assertNoLogs(self.logger, level="WARNING"):
            self.assertEqual(reap_process_groups(self.root, logger=self.logger), 0)
        self.assertIsNone(process.poll())
        self.assertFalse(path.exists())

    def test_a_group_whose_leader_already_exited_is_not_guessed_at(self):
        path = self.root / ("call-gone" + PGID_SUFFIX)
        path.write_text("999999999 424242\n")
        self.assertEqual(reap_process_groups(self.root, logger=self.logger), 0)
        self.assertFalse(path.exists())

    def test_records_are_found_under_every_role_home(self):
        nested = self.root / "conversation/assistant/scope"
        nested.mkdir(parents=True)
        process, _ = self.group()
        record_process_group(nested / ("call-e1" + PGID_SUFFIX), process.pid)
        self.assertEqual(reap_process_groups(self.root, logger=self.logger), 1)
        process.wait(timeout=5)

    def test_an_unreadable_record_is_reported_and_discarded(self):
        path = self.root / ("call-bad" + PGID_SUFFIX)
        path.write_text("not a process group\n")
        with self.assertLogs(self.logger, level="WARNING"):
            self.assertEqual(reap_process_groups(self.root, logger=self.logger), 0)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
