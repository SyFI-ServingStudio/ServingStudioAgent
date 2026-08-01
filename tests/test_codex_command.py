import unittest

from backend.codex_runtime.codex_command import build_codex_exec_command
from backend.codex_runtime.config import role_codex_home_in_container
from backend.codex_runtime.exec_types import CodexExecRequest
from backend.managed_context import MANAGED_CONTEXT_CONTAINER_PATH


def _request(
    *, session_id: str | None = None, backend_id: str = "traditional"
) -> CodexExecRequest:
    return CodexExecRequest(
        container="test-container",
        prompt="question",
        label="orchestrator",
        workspace_id="w_main",
        conversation_id="conversation",
        turn_id="turn",
        session_id=session_id,
        backend_id=backend_id,
    )


class CodexCommandTests(unittest.TestCase):
    def test_codexds_uses_isolated_codex_home_and_profile_model(self) -> None:
        command = build_codex_exec_command(_request(backend_id="codexds"))

        self.assertIn(
            f"CODEX_HOME={role_codex_home_in_container('orchestrator')}", command
        )
        self.assertNotIn("-m", command)
        self.assertNotIn("model_reasoning_effort=xhigh", command)

    def test_traditional_preserves_explicit_model_and_reasoning_options(self) -> None:
        command = build_codex_exec_command(_request(backend_id="traditional"))

        self.assertIn("-m", command)
        self.assertIn("gpt-5.6-sol", command)
        self.assertIn('model_reasoning_effort="xhigh"', command)

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
