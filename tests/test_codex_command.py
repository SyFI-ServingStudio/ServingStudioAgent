import unittest

from backend.codex_runtime.codex_command import build_codex_exec_command
from backend.codex_runtime.config import CODEXDS_MODEL, role_codex_home_in_container
from backend.codex_runtime.exec_types import CodexExecRequest
from backend.managed_context import MANAGED_CONTEXT_CONTAINER_PATH


def _request(
    *,
    session_id: str | None = None,
    model_id: str = "gpt-5.6-sol",
    effort: str = "xhigh",
    output_schema: str | None = None,
) -> CodexExecRequest:
    return CodexExecRequest(
        container="test-container",
        prompt="question",
        label="orchestrator",
        workspace_id="w_main",
        conversation_id="conversation",
        turn_id="turn",
        session_id=session_id,
        model_id=model_id,
        effort=effort,
        output_schema=output_schema,
    )


class CodexCommandTests(unittest.TestCase):
    def test_deepseek_gets_an_explicit_model_and_effort(self) -> None:
        """Neither may fall back to the profile's config.toml default.

        The DeepSeek profile pins both in its own `config.toml`, so before the
        registry it worked by accident — and its effort could not be changed.
        """
        command = build_codex_exec_command(
            _request(model_id=CODEXDS_MODEL, effort="high")
        )

        self.assertIn(
            f"CODEX_HOME={role_codex_home_in_container('orchestrator')}", command
        )
        self.assertIn("-m", command)
        self.assertIn(CODEXDS_MODEL, command)
        self.assertIn('model_reasoning_effort="high"', command)

    def test_gpt_family_carries_the_selected_model_and_effort(self) -> None:
        command = build_codex_exec_command(
            _request(model_id="gpt-5.6-luna", effort="medium")
        )

        self.assertIn("-m", command)
        self.assertIn("gpt-5.6-luna", command)
        self.assertIn('model_reasoning_effort="medium"', command)

    def test_resume_keeps_model_and_effort_options(self) -> None:
        """A within-family model switch happens on a resumed session."""
        command = build_codex_exec_command(
            _request(session_id="session-1", model_id="gpt-5.6-terra", effort="high")
        )

        self.assertIn("resume", command)
        self.assertIn("gpt-5.6-terra", command)
        self.assertIn('model_reasoning_effort="high"', command)

    def test_fresh_and_resumed_orchestrators_share_the_output_schema(self) -> None:
        schema = "/opt/vibesim/prompts/orchestrator.schema.json"
        fresh = build_codex_exec_command(_request(output_schema=schema))
        resumed = build_codex_exec_command(
            _request(session_id="session-1", output_schema=schema)
        )

        for command in (fresh, resumed):
            self.assertIn("--output-schema", command)
            self.assertIn(schema, command)

    def test_new_codex_call_injects_analyzer_mcp(self) -> None:
        command = build_codex_exec_command(_request())
        self.assertIn(
            'mcp_servers.analyzer.command="/opt/vibesim-analyzer-mcp-venv/bin/python"',
            command,
        )
        self.assertTrue(
            any(
                option.startswith(
                    'mcp_servers.analyzer.args=["/opt/vibesim/analyzer-evidence-mcp/'
                )
                for option in command
            )
        )
        self.assertIn(
            'mcp_servers.analyzer.env.ANALYZER_MCP_SOURCE="external"',
            command,
        )
        self.assertIn(
            "mcp_servers.analyzer.env.ANALYZER_MCP_BASE_URL="
            '"http://host.docker.internal:8787"',
            command,
        )
        self.assertIn(
            "mcp_servers.analyzer.env.VIBESIM_MANAGED_RUN_CONTEXT="
            f'"{MANAGED_CONTEXT_CONTAINER_PATH}"',
            command,
        )
        self.assertIn(
            f"VIBESIM_MANAGED_JOB_CONTEXT={MANAGED_CONTEXT_CONTAINER_PATH}",
            command,
        )

    def test_resumed_codex_call_also_injects_analyzer_mcp(self) -> None:
        command = build_codex_exec_command(_request(session_id="session"))
        self.assertIn(
            'mcp_servers.analyzer.command="/opt/vibesim-analyzer-mcp-venv/bin/python"',
            command,
        )
