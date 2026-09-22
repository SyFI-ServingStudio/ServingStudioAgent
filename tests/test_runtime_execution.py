import asyncio
import json
import logging
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.runtime_fixtures import agent_request, docker_execution, execution_environment
from vibesim_agent.runtime.execution import PID_WRAPPER, DockerExecution, HostExecution
from vibesim_agent.runtime.host import reap_process_groups
from vibesim_agent.runtime.invocation import STOP_TIMEOUT, HostInvocation, InvocationHome
from vibesim_agent.runtime.permissions import host_permissions
from vibesim_agent.runtime.process import kill_process_group


class DockerExecutionTests(unittest.TestCase):
    """The seam must reproduce the transport it replaced, argument for argument.

    The golden command fixtures are the wider proof; these pin the pieces the
    fixtures cannot isolate, so a regression names itself instead of showing up
    as a diff in a 70-element list.
    """

    def setUp(self):
        self.environment = execution_environment()
        self.execution = DockerExecution(self.environment, "container-id")

    def test_prefix_is_the_execution_environment_prefix(self):
        self.assertEqual(
            self.execution.prefix(environment={"CODEX_HOME": "/home"}),
            self.environment.prefix("container-id", environment={"CODEX_HOME": "/home"}),
        )

    def test_command_appends_the_pid_wrapper_between_prefix_and_argv(self):
        command = self.execution.command(
            ["codex", "exec"],
            environment={"CODEX_HOME": "/home"},
            pid_file="/home/call-1.pid",
            label="vibesim-codex",
        )
        prefix = self.execution.prefix(environment={"CODEX_HOME": "/home"})
        self.assertEqual(
            command,
            [*prefix, "sh", "-c", PID_WRAPPER, "vibesim-codex", "/home/call-1.pid", "codex", "exec"],
        )

    def test_the_wrapper_refuses_to_start_once_cancellation_is_marked(self):
        # The remote shell can start after its local client exits; this check is
        # the only thing standing between that and a turn nobody is watching.
        self.assertIn('if [ -e "$1.cancel" ]', PID_WRAPPER)
        self.assertIn("exit 130", PID_WRAPPER)

    def test_container_execution_reports_no_working_directory(self):
        self.assertIsNone(self.execution.cwd)
        self.assertEqual(self.execution.agent_prompt, "/workspace/AGENTS.md")
        self.assertEqual(self.execution.stop_timeout, STOP_TIMEOUT)
        self.assertEqual(self.execution.managed_context, self.environment.managed_context)
        self.assertEqual(
            self.execution.home_path(InvocationHome(Path("/host"), "/in-container")),
            "/in-container",
        )

    def test_spawn_environment_is_passed_through_untouched(self):
        # `docker exec -e NAME` inherits by name from the client process, so the
        # credentials have to reach the spawn unchanged.
        self.assertIsNone(self.execution.spawn_environment(None))
        self.assertEqual(
            self.execution.spawn_environment({"ANTHROPIC_API_KEY": "secret"}),
            {"ANTHROPIC_API_KEY": "secret"},
        )

    def test_inherited_names_that_collide_with_runtime_values_are_refused(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.execution.command(
                ["codex"],
                environment={},
                inherited=("USER",),
                pid_file="/p",
                label="vibesim-codex",
            )


class ManagedContextSeamTests(unittest.TestCase):
    """The MCP context path comes from the turn, not from the command builder.

    It is a process-global constant in a container and per-conversation on the
    host, so reading it off the builder's frozen environment would be wrong in
    exactly one of the two modes.
    """

    def test_codex_takes_the_context_from_the_execution(self):
        from vibesim_agent.providers.codex.command import CodexCommand

        builder = CodexCommand(execution_environment())
        request = replace(
            agent_request(),
            execution=docker_execution(
                replace(execution_environment(), managed_context="/per/turn.json")
            ),
        )
        command = builder.build(request, home="/home")
        settings = [
            command[index + 1] for index, part in enumerate(command) if part == "-c"
        ]
        self.assertIn(
            'mcp_servers.analyzer.env.VIBESIM_MANAGED_RUN_CONTEXT="/per/turn.json"',
            settings,
        )

    def test_claude_takes_the_context_and_prompt_path_from_the_execution(self):
        from vibesim_agent.providers.claude.command import ClaudeCommand

        root = Path(__file__).parents[1] / "vibesim_agent/prompts/contracts"
        builder = ClaudeCommand(
            execution_environment(),
            {path.name: json.loads(path.read_text()) for path in root.glob("*.schema.json")},
        )
        execution = docker_execution(
            replace(execution_environment(), managed_context="/per/turn.json")
        )
        request = replace(
            agent_request(),
            execution=execution,
            output_schema=Path("/contracts/assistant.schema.json"),
        )
        arguments = builder.arguments(request)
        mcp = json.loads(arguments[arguments.index("--mcp-config") + 1])
        self.assertEqual(
            mcp["mcpServers"]["analyzer"]["env"]["VIBESIM_MANAGED_RUN_CONTEXT"],
            "/per/turn.json",
        )
        self.assertEqual(
            arguments[arguments.index("--append-system-prompt-file") + 1],
            execution.agent_prompt,
        )


class HostExecutionTests(unittest.TestCase):
    def execution(self, **overrides):
        return HostExecution(
            **{
                "repo": Path("/trees/wt-topic"),
                "agent_prompt": "/state/prompts/AGENTS.md",
                "managed_context": "/state/c/managed/context.json",
                "analyzer_source": "external",
                "analyzer_base_url": "http://172.17.0.1:63044",
                "managed_backend_url": "http://172.17.0.1:63043",
                "mcp_python": "/trees/wt-topic/.venv/bin/python",
                "mcp_server": "/agent/vibesim_agent/analyzer_evidence_mcp/server.py",
                "permissions": host_permissions(
                    git_dir=Path("/trees/wt-topic/.git"),
                    git_common_dir=Path("/base/.git"),
                ),
                **overrides,
            }
        )

    def test_command_assigns_the_turn_variables_and_execs_without_a_shell(self):
        execution = self.execution()
        command = execution.command(
            ["codex", "exec", "--json"],
            environment={"CODEX_HOME": "/state/home"},
            pid_file="/state/home/call-1.pid",
            label="vibesim-codex",
        )
        # The pid wrapper answers a `docker exec` race that does not exist here,
        # and a shell would mean quoting argv that carries whole JSON schemas.
        self.assertNotIn("sh", command)
        self.assertNotIn("docker", command)
        self.assertEqual(command[0], "env")
        self.assertEqual(command[-3:], ["codex", "exec", "--json"])
        assignments = dict(part.split("=", 1) for part in command[1:-3])
        # Without CODEX_HOME the CLI would silently fall back to the operator's
        # own `~/.codex` rather than the role home this turn prepared.
        self.assertEqual(assignments["CODEX_HOME"], "/state/home")
        self.assertEqual(
            assignments["ANALYZER_MCP_BASE_URL"], execution.analyzer_base_url
        )
        self.assertEqual(
            assignments["VIBESIM_MANAGED_RUN_CONTEXT"], execution.managed_context
        )
        self.assertEqual(
            assignments["VIBESIM_MANAGED_JOB_CONTEXT"], execution.managed_context
        )

    def test_the_assigned_command_actually_runs_with_those_variables(self):
        # `env` is in the argv, so it has to be the real one: a typo here would
        # only show up as a CLI that cannot find its role home.
        execution = self.execution()
        command = execution.command(
            [sys.executable, "-c", "import os;print(os.environ['CODEX_HOME'])"],
            environment={"CODEX_HOME": "/state/home"},
            pid_file="/unused",
            label="vibesim-codex",
        )
        output = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(output.stdout.strip(), "/state/home")

    def test_conflicting_inherited_names_are_still_refused(self):
        for name in ("CODEX_HOME", "ANALYZER_MCP_BASE_URL", "VIBESIM_MANAGED_RUN_CONTEXT"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    self.execution().command(
                        ["codex"],
                        environment={"CODEX_HOME": "/state/home"},
                        inherited=(name,),
                        pid_file="/p",
                        label="vibesim-codex",
                    )

    def test_container_only_variables_are_dropped_from_the_spawn(self):
        spawned = self.execution().spawn_environment(
            {
                "PATH": "/usr/bin",
                "ANTHROPIC_API_KEY": "secret",
                "HOME": "/home/runner",
                "UV_PROJECT_ENVIRONMENT": "/opt/vibesim-venv",
                "UV_CACHE_DIR": "/opt/vibesim-uv-cache",
                "VIBESIM_EXPECTED_LOCK_SHA": "sha",
                "VIBESIM_RUNNER_GPUS": "all",
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                "DG_USE_LOCAL_VERSION": "0",
            }
        )
        # These name paths inside the runner image. Inheriting either uv
        # variable makes `uv run` in the worktree fail outright.
        self.assertEqual(spawned, {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "secret"})
        self.assertIsNone(self.execution().spawn_environment(None))

    def test_the_group_is_what_gets_killed(self):
        execution = self.execution()
        self.assertTrue(execution.start_new_session)
        self.assertIs(execution.kill, kill_process_group)
        self.assertEqual(execution.cwd, Path("/trees/wt-topic"))

    def test_role_home_is_the_host_path(self):
        self.assertEqual(
            self.execution().home_path(InvocationHome(Path("/host"), "/in-container")),
            "/host",
        )

    def test_a_relative_repository_is_refused(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            self.execution(repo=Path("wt-topic"))


class HostInvocationTests(unittest.IsolatedAsyncioTestCase):
    def invocation(self, directory, **overrides):
        return HostInvocation(
            execution_id="e1",
            home=InvocationHome(Path(directory), str(directory)),
            logger=logging.getLogger("host-invocation-test"),
            label="Codex",
            **overrides,
        )

    async def test_stop_reaps_a_grandchild_the_leader_would_orphan(self):
        with TemporaryDirectory() as directory:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-c", GRANDCHILD,
                stdout=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            grandchild = int((await process.stdout.readline()).strip())
            await self.invocation(directory, signal_grace=0.2).stop(process)

            self.assertIsNotNone(process.returncode)
            with self.assertRaises(ProcessLookupError):
                os.kill(grandchild, 0)

    async def test_an_already_dead_process_stops_without_error(self):
        with TemporaryDirectory() as directory:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", "pass", start_new_session=True
            )
            await process.wait()
            await self.invocation(directory, signal_grace=0.2).stop(process)

    async def test_mark_cancelled_writes_nothing(self):
        with TemporaryDirectory() as directory:
            invocation = self.invocation(directory)
            invocation.mark_cancelled()
            # There is no window to guard: create_subprocess_exec returns only
            # after fork and exec have both succeeded.
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(invocation.pid_file, str(Path(directory) / "call-e1.pid"))


GRANDCHILD = (
    "import subprocess,sys,time;"
    "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
    "print(child.pid, flush=True);"
    "time.sleep(30)"
)


class ProcessGroupRecordTests(unittest.IsolatedAsyncioTestCase):
    """The record is written by the running turn and read by the next startup."""

    async def test_the_record_survives_the_process_and_is_removed_with_it(self):
        with TemporaryDirectory() as directory:
            home = InvocationHome(Path(directory), str(directory))
            invocation = HostInvocation(
                execution_id="e1",
                home=home,
                logger=logging.getLogger("process-group-record-test"),
                label="Codex",
                signal_grace=0.2,
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", "import time;time.sleep(30)",
                start_new_session=True,
            )
            invocation.record(process)
            recorded = Path(directory) / "call-e1.pgid"
            pgid, started = recorded.read_text().split()
            self.assertEqual(int(pgid), os.getpgid(process.pid))
            self.assertTrue(started.isdigit())
            # A backend that died here would find this file and could act on it.
            self.assertEqual(
                reap_process_groups(
                    Path(directory), logger=logging.getLogger("reap-test")
                ),
                1,
            )
            await process.wait()

            # Recording a process that is already gone writes nothing rather
            # than a number that now belongs to nobody.
            with self.assertLogs("process-group-record-test", level="WARNING"):
                invocation.record(process)
            self.assertFalse(recorded.exists())

            recorded.write_text("1 1\n")
            invocation.remove_pid()
            # Removed on the ordinary path too: the turn is over, and the number
            # is free to be handed to something else.
            self.assertFalse(recorded.exists())


if __name__ == "__main__":
    unittest.main()
