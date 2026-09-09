import unittest
from unittest.mock import patch

from backend.codex_runtime import config
from backend.codex_runtime.codex_command import build_codex_exec_command
from backend.codex_runtime.config import (
    ASSISTANT_SCHEMA_IN_CONTAINER,
    CODEXDS_MODEL,
    ORCHESTRATOR_SCHEMA_IN_CONTAINER,
    role_codex_home_in_container,
)
from backend.codex_runtime.exec_types import CodexExecRequest
from backend.managed_context import MANAGED_CONTEXT_CONTAINER_PATH
from model_catalog_fixture import install_model_catalog


def _request(
    *,
    session_id: str | None = None,
    model_id: str = "gpt-5.6-sol",
    effort: str = "xhigh",
    service_tier: str = "default",
    output_schema: str | None = None,
    label: str = "orchestrator",
) -> CodexExecRequest:
    return CodexExecRequest(
        container="test-container",
        prompt="question",
        label=label,
        workspace_id="w_main",
        conversation_id="conversation",
        turn_id="turn",
        session_id=session_id,
        model_id=model_id,
        effort=effort,
        service_tier=service_tier,
        output_schema=output_schema,
    )


class CodexCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        install_model_catalog(self)

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
            _request(
                model_id="gpt-5.6-luna", effort="medium", service_tier="fast"
            )
        )

        self.assertIn("-m", command)
        self.assertIn("gpt-5.6-luna", command)
        self.assertIn('model_reasoning_effort="medium"', command)
        self.assertIn('service_tier="fast"', command)

    def test_missing_catalog_does_not_enable_fast_service(self) -> None:
        with (
            patch.object(config, "_catalog_models", return_value={}),
            patch.object(config, "_MODEL_REGISTRY_CACHE", {}),
        ):
            command = build_codex_exec_command(
                _request(model_id="gpt-5.6-luna", service_tier="fast")
            )
        self.assertFalse(any("service_tier=" in option for option in command))

    def test_deepseek_does_not_receive_an_unsupported_service_tier(self) -> None:
        command = build_codex_exec_command(
            _request(model_id=CODEXDS_MODEL, effort="max", service_tier="default")
        )

        self.assertFalse(any("service_tier=" in option for option in command))

    def test_resume_keeps_model_and_effort_options(self) -> None:
        """A within-family model switch happens on a resumed session."""
        command = build_codex_exec_command(
            _request(
                session_id="session-1",
                model_id="gpt-5.6-terra",
                effort="high",
                service_tier="fast",
            )
        )

        self.assertIn("resume", command)
        self.assertIn("gpt-5.6-terra", command)
        self.assertIn('model_reasoning_effort="high"', command)
        self.assertIn('service_tier="fast"', command)

    def test_fresh_and_resumed_drivers_share_the_output_schema(self) -> None:
        """Both agent modes, both call shapes: the schema is never dropped."""
        for label, schema in (
            ("orchestrator", ORCHESTRATOR_SCHEMA_IN_CONTAINER),
            ("assistant", ASSISTANT_SCHEMA_IN_CONTAINER),
        ):
            fresh = build_codex_exec_command(
                _request(label=label, output_schema=schema)
            )
            resumed = build_codex_exec_command(
                _request(label=label, session_id="session-1", output_schema=schema)
            )

            for command in (fresh, resumed):
                with self.subTest(label=label, resumed="resume" in command):
                    self.assertIn("--output-schema", command)
                    self.assertIn(schema, command)
                    self.assertIn(
                        f"CODEX_HOME={role_codex_home_in_container(label)}", command
                    )

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
