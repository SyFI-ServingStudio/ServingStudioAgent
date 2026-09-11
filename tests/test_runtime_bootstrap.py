"""Execute bootstrap/readiness protocols in a temporary local filesystem."""

import os
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vibesim_agent.runtime import bootstrap
from vibesim_agent.runtime.command import ExecutionEnvironment
from vibesim_agent.settings import AgentSettings, ContainerSettings


class RuntimeBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.home = self.root / "home ' ; $(touch PWNED)"
        self.workspace = self.root / "workspace"
        self.mcp = self.root / "mcp"
        self.seed = self.root / "seed"
        self.uv = self.root / "uv"
        self.role = self.root / "role ' home"
        self.hf = self.root / "hf"
        self.bin = self.root / "bin"
        for directory in (
            self.home,
            self.workspace,
            self.mcp,
            self.seed / "release",
            self.uv,
            self.role,
            self.hf,
            self.bin,
        ):
            directory.mkdir(parents=True)
        self.marker = self.root / "ready"
        self.prompt = self.root / "prompt ' selected.md"
        self.prompt.write_text("selected instructions")
        (self.workspace / "AGENTS.md").write_text("selected instructions")
        (self.mcp / "server.py").write_text("pass")
        (self.seed / "release" / "simulator").write_text("prebuilt simulator")
        for tool in (*bootstrap.BASE_BINARIES, "claude", "nvidia-smi"):
            path = self.bin / tool
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
        settings = ContainerSettings(
            image="runner",
            uid=os.getuid(),
            gid=os.getgid(),
            user="runner",
            home=self.home,
            gpus="",
            uv_project_environment=self.uv,
        )
        agent = AgentSettings(
            repo_root=self.root, main_dir=self.workspace, workspaces_root=self.root
        )
        self.environment = ExecutionEnvironment(settings, agent, "lock-sha", "context")
        self.variables = {
            **os.environ,
            "PATH": str(self.bin) + ":/usr/bin:/bin",
            "HOME": str(self.home),
            "USER": "runner",
            "LOGNAME": "runner",
            "DG_USE_LOCAL_VERSION": "0",
            "VIBESIM_BAKED_LOCK_SHA": "lock-sha",
            "VIBESIM_BAKED_TARGET": str(self.seed),
            "UV_PROJECT_ENVIRONMENT": str(self.uv),
            "ANALYZER_MCP_SOURCE": agent.analyzer_source,
            "ANALYZER_MCP_BASE_URL": agent.analyzer_base_url,
        }
        for name, value in (
            ("READY_MARKER", str(self.marker)),
            ("WORKSPACE_TARGET", self.workspace),
            ("MCP_TARGET", self.mcp),
            ("HF_TARGET", self.hf),
        ):
            self.enterContext(patch.object(bootstrap, name, value))

    def script(self, ready=False, environment=None, **kwargs):
        options = {
            "fingerprint": "fingerprint",
            "binaries": ("claude",),
            "role_homes": (str(self.role),),
            "agent_prompt": str(self.prompt),
        }
        options.update(kwargs)
        build = bootstrap.readiness_script if ready else bootstrap.bootstrap_script
        return build(environment or self.environment, **options)

    def run_script(self, ready=False, environment=None, variables=None, **kwargs):
        return subprocess.run(
            ["/bin/bash", "-c", self.script(ready, environment, **kwargs)],
            env={**self.variables, **(variables or {})},
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    def test_bootstrap_seeds_once_publishes_fingerprint_and_readiness_is_read_only(
        self,
    ):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        target = self.workspace / "target" / "release" / "simulator"
        self.assertEqual(target.read_text(), "prebuilt simulator")
        self.assertEqual(self.marker.read_text(), "fingerprint\n")
        target.write_text("existing workspace artifact")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(target.read_text(), "existing workspace artifact")
        before = {str(p): p.stat().st_mtime_ns for p in self.root.rglob("*")}
        result = self.run_script(ready=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            {str(p): p.stat().st_mtime_ns for p in self.root.rglob("*")}, before
        )
        self.assertFalse((self.root / "PWNED").exists())

    def test_failed_bootstrap_removes_stale_marker_for_environment_mismatches(self):
        cases = (
            {"DG_USE_LOCAL_VERSION": "1"},
            {"VIBESIM_BAKED_LOCK_SHA": "wrong"},
            {"VIBESIM_BAKED_TARGET": ""},
            {"UV_PROJECT_ENVIRONMENT": str(self.root)},
            {"HOME": "/wrong"},
            {"USER": "wrong"},
            {"LOGNAME": "wrong"},
            {"ANALYZER_MCP_SOURCE": "wrong"},
            {"ANALYZER_MCP_BASE_URL": "http://wrong"},
        )
        for variables in cases:
            with self.subTest(variables=variables):
                self.marker.write_text("old fingerprint")
                self.assertNotEqual(self.run_script(variables=variables).returncode, 0)
                self.assertFalse(self.marker.exists())
        self.assertFalse((self.workspace / "target").exists())

    def test_uid_and_gid_mismatch_reject_before_ready(self):
        for key in ("uid", "gid"):
            settings = self.environment.container
            environment = replace(
                self.environment,
                container=settings.model_copy(update={key: getattr(settings, key) + 1}),
            )
            result = self.run_script(environment=environment)
            self.assertEqual(result.returncode, 126)
            self.assertFalse(self.marker.exists())

    def test_missing_roles_tools_prompt_mcp_and_seed_fail(self):
        for options in (
            {"role_homes": (str(self.root / "missing"),)},
            {"binaries": ("missing-provider-binary",)},
            {"agent_prompt": str(self.root / "missing")},
        ):
            with self.subTest(options=options):
                self.assertNotEqual(self.run_script(**options).returncode, 0)
                self.assertFalse(self.marker.exists())
        (self.workspace / "AGENTS.md").write_text("wrong prompt")
        self.assertNotEqual(self.run_script().returncode, 0)
        (self.workspace / "AGENTS.md").write_text(self.prompt.read_text())
        (self.mcp / "server.py").unlink()
        self.assertNotEqual(self.run_script().returncode, 0)
        (self.mcp / "server.py").write_text("pass")
        self.assertNotEqual(
            self.run_script(
                variables={"VIBESIM_BAKED_TARGET": str(self.root / "missing")}
            ).returncode,
            0,
        )
        self.assertFalse(self.marker.exists())

    def test_provider_binaries_are_explicit_and_values_are_shell_quoted(self):
        script = self.script(binaries=("claude",))
        self.assertNotIn("codex", script)
        self.assertNotIn("cargo build", script)
        self.assertNotIn("uv sync", script)
        fingerprint = "literal'; touch PWNED; #"
        result = self.run_script(fingerprint=fingerprint)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.marker.read_text(), fingerprint + "\n")
        self.assertFalse((self.root / "PWNED").exists())
        self.assertEqual(
            self.run_script(ready=True, fingerprint=fingerprint).returncode, 0
        )
        self.assertNotEqual(
            self.run_script(ready=True, fingerprint="different").returncode, 0
        )

    def test_gpu_and_hf_are_conditional_and_validate_actual_runtime(self):
        environment = replace(
            self.environment,
            container=self.environment.container.model_copy(
                update={"gpus": "all", "hf_home": self.root / "host-cache"}
            ),
        )
        self.assertNotEqual(self.run_script(environment=environment).returncode, 0)
        result = self.run_script(
            environment=environment, variables={"HF_HOME": str(self.hf)}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.bin / "nvidia-smi").write_text("#!/bin/sh\nexit 1\n")
        self.assertNotEqual(
            self.run_script(
                ready=True, environment=environment, variables={"HF_HOME": str(self.hf)}
            ).returncode,
            0,
        )
        self.assertEqual(self.run_script().returncode, 0)

    def test_readiness_rechecks_prompt_and_target_instead_of_trusting_marker(self):
        self.assertEqual(self.run_script().returncode, 0)
        (self.workspace / "AGENTS.md").write_text("changed")
        self.assertNotEqual(self.run_script(ready=True).returncode, 0)
        (self.workspace / "AGENTS.md").write_text(self.prompt.read_text())
        (self.workspace / "target" / "release" / "simulator").unlink()
        (self.workspace / "target" / "release").rmdir()
        (self.workspace / "target").rmdir()
        self.assertNotEqual(self.run_script(ready=True).returncode, 0)
        self.assertFalse((self.workspace / "target").exists())

    def test_copy_failure_cannot_publish_ready_marker(self):
        copy = self.bin / "cp"
        copy.write_text("#!/bin/sh\nexit 17\n")
        copy.chmod(0o755)
        self.marker.write_text("stale")
        result = self.run_script()
        self.assertEqual(result.returncode, 17)
        self.assertFalse(self.marker.exists())
        self.assertFalse((self.workspace / "target").exists())
        self.assertEqual(list(self.workspace.glob(".vibesim-target.*")), [])
        copy.unlink()
        self.assertEqual(self.run_script().returncode, 0)

    def test_existing_empty_or_debug_only_target_is_preserved(self):
        target = self.workspace / "target"
        target.mkdir()
        for debug in (False, True):
            if debug:
                (target / "debug").mkdir()
                (target / "debug" / "simulator").write_text("debug artifact")
            before = {str(p): p.stat().st_mtime_ns for p in target.rglob("*")}
            result = self.run_script()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.run_script(ready=True).returncode, 0)
            self.assertEqual(
                {str(p): p.stat().st_mtime_ns for p in target.rglob("*")}, before
            )
            self.assertFalse((target / "release").exists())

    def test_concurrent_target_created_during_copy_wins_without_overwrite(self):
        copy = self.bin / "cp"
        copy.write_text(
            '#!/bin/sh\n/bin/cp "$@"\nmkdir -p target/debug\nprintf concurrent > target/debug/owner\n'
        )
        copy.chmod(0o755)
        self.variables["PWD"] = str(self.workspace)
        result = subprocess.run(
            ["/bin/bash", "-c", self.script()],
            env=self.variables,
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.workspace / "target" / "debug" / "owner").read_text(), "concurrent"
        )
        self.assertFalse((self.workspace / "target" / "release").exists())
        self.assertEqual(list(self.workspace.glob(".vibesim-target.*")), [])
