"""Host PATH readiness: a hard failure for absence, a warning for drift."""

import logging
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.runtime.host import (
    PINNED_VERSIONS,
    HostUnavailable,
    check_host_binaries,
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


if __name__ == "__main__":
    unittest.main()
