import unittest

from backend.codex_runtime.codex_command import build_codex_exec_command
from backend.codex_runtime.exec_types import CodexExecRequest
from backend.managed_context import MANAGED_CONTEXT_CONTAINER_PATH


def _request(*, session_id: str | None = None) -> CodexExecRequest:
    return CodexExecRequest(
        container="test-container",
        prompt="question",
        label="orchestrator",
        workspace_id="w_main",
        conversation_id="conversation",
        turn_id="turn",
        session_id=session_id,
    )


class CodexCommandTests(unittest.TestCase):
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
