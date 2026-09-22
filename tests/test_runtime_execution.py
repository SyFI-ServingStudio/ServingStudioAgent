import json
import unittest
from dataclasses import replace
from pathlib import Path

from tests.runtime_fixtures import agent_request, docker_execution, execution_environment
from vibesim_agent.runtime.execution import PID_WRAPPER, DockerExecution
from vibesim_agent.runtime.invocation import STOP_TIMEOUT, InvocationHome


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


if __name__ == "__main__":
    unittest.main()
