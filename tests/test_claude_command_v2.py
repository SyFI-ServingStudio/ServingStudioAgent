import json
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tests import test_codex_command_v2 as codex_fixture
from vibesim_agent.domain.roles import Role
from vibesim_agent.providers.base import OutputMode
from vibesim_agent.providers.claude.command import ClaudeCommand


class ClaudeCommandTests(unittest.TestCase):
    def builder(self):
        root = Path(__file__).parents[1] / "vibesim_agent/prompts/contracts"
        return ClaudeCommand(
            codex_fixture.CodexCommandTests().builder().environment,
            {
                path.name: json.loads(path.read_text())
                for path in root.glob("*.schema.json")
            },
            inherited_environment=tuple(
                codex_fixture.command_golden()["inherited_claude_environment"]
            ),
        )

    def request(self):
        request = codex_fixture.CodexCommandTests().request()
        return replace(request, output_schema=Path("/contracts/assistant.schema.json"))

    def test_cli_arguments_match_legacy_for_all_roles_and_resume(self):
        builder = self.builder()
        for role in Role:
            for session in (None, "saved-session"):
                with self.subTest(role=role, session=session):
                    request = replace(
                        self.request(),
                        role=role,
                        session_id=session,
                        output_schema=Path(f"/contracts/{role.value}.schema.json"),
                    )
                    case = next(
                        case
                        for case in codex_fixture.command_golden()["cases"]
                        if case["role"] == role.value and case["session_id"] == session
                    )
                    home = case["home"] + "/claude"
                    pid = home + f"/call-{request.execution_id}.pid"
                    actual = builder.build_tracked(request, home=home, pid_file=pid)
                    legacy = case["claude"]

                    def split_prefix(command):
                        prefix = iter(command[: command.index("sh")])
                        environment, arguments = [], []
                        for arg in prefix:
                            if arg == "-e":
                                environment.append(next(prefix))
                            else:
                                arguments.append(arg)
                        return sorted(environment), arguments

                    self.assertEqual(
                        split_prefix(actual),
                        split_prefix(
                            codex_fixture.normalize_legacy_gpu_environment(legacy)
                        ),
                    )
                    self.assertEqual(
                        actual[actual.index("-p") :], legacy[legacy.index("-p") :]
                    )
                    self.assertNotIn(request.prompt, actual)

    def test_schema_dialect_is_removed_only_from_cli_copy(self):
        builder = self.builder()
        request = self.request()
        original = builder.output_schema(request)
        command = builder.build_tracked(
            request, home="/home", pid_file="/home/call.pid"
        )
        transmitted = json.loads(command[command.index("--json-schema") + 1])
        self.assertNotIn("$schema", transmitted)
        self.assertIn("$schema", original)
        self.assertEqual(builder.output_schema(request), original)
        original.clear()
        self.assertTrue(builder.output_schema(request))

    def test_capability_controls_schema_and_unknown_contract_rejected(self):
        request = self.request()
        builder = self.builder()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            builder.output_schema(replace(request, output_schema=Path("/unknown.json")))
        request = replace(
            request,
            selection=replace(
                request.selection,
                model=replace(request.selection.model, output_mode=OutputMode.PROMPT),
            ),
        )
        self.assertIsNone(builder.output_schema(request))
        self.assertNotIn(
            "--json-schema",
            builder.build_tracked(request, home="/home", pid_file="/home/call.pid"),
        )

    def test_credentials_are_inherited_by_name_and_mcp_values_are_json(self):
        builder = replace(
            self.builder(),
            inherited_environment=("ANTHROPIC_API_KEY",),
            mcp_python='/space/"python',
            mcp_server="/space dir/server.py",
        )
        command = builder.build_tracked(
            self.request(), home="/home", pid_file="/home/call.pid"
        )
        self.assertIn("ANTHROPIC_API_KEY", command)
        self.assertFalse(any(arg.startswith("ANTHROPIC_API_KEY=") for arg in command))
        mcp = json.loads(command[command.index("--mcp-config") + 1])["mcpServers"][
            "analyzer"
        ]
        self.assertEqual(mcp["command"], builder.mcp_python)
        self.assertEqual(mcp["args"], [builder.mcp_server])

    def test_tracked_shell_honors_cancellation_before_executing(self):
        with TemporaryDirectory() as directory:
            pid = Path(directory) / "call with spaces.pid"
            marker = pid.with_suffix(".pid.cancel")
            command = self.builder().build_tracked(
                self.request(), home=directory, pid_file=str(pid)
            )
            shell = command[command.index("sh") : command.index("claude")]
            marker.touch()
            result = subprocess.run(
                [*shell, "sh", "-c", "exit 23"], check=False, timeout=5
            )
            self.assertEqual(result.returncode, 130)
            self.assertTrue(marker.exists())
            self.assertFalse(pid.exists())
            marker.unlink()
            result = subprocess.run(
                [*shell, "sh", "-c", "exit 23"], check=False, timeout=5
            )
            self.assertEqual(result.returncode, 23)
            self.assertTrue(pid.read_text().strip().isdigit())
