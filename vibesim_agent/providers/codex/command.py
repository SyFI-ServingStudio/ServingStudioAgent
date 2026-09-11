"""Codex command construction without configuration or filesystem reads."""

import json
from dataclasses import dataclass
from pathlib import PurePosixPath

from ...runtime.command import ExecutionEnvironment
from ..base import AgentRequest


@dataclass(frozen=True)
class CodexCommand:
    environment: ExecutionEnvironment
    mcp_python: str = "/opt/vibesim-analyzer-mcp-venv/bin/python"
    mcp_server: str = "/opt/vibesim/analyzer-evidence-mcp/server.py"
    catalog_filename: str | None = None
    inherited_environment: tuple[str, ...] = ()

    def __post_init__(self):
        if self.catalog_filename not in {None, "models_catalog.json"}:
            raise ValueError("unsupported Codex catalog filename")

    def build_tracked(
        self, request: AgentRequest, *, home: str, pid_file: str
    ) -> list[str]:
        prefix = self.environment.prefix(
            request.container,
            environment={"CODEX_HOME": home},
            inherited=self.inherited_environment,
        )
        arguments = self.build(request, home=home)[len(prefix) :]
        return [
            *prefix,
            "sh",
            "-c",
            (
                'printf "%s\\n" "$$" > "$1" || exit; '
                'if [ -e "$1.cancel" ]; then rm -f -- "$1"; exit 130; fi; '
                'shift; exec "$@"'
            ),
            "vibesim-codex",
            pid_file,
            *arguments,
        ]

    def build(self, request: AgentRequest, *, home: str) -> list[str]:
        selection = request.selection
        command = self.environment.prefix(
            request.container,
            environment={"CODEX_HOME": home},
            inherited=self.inherited_environment,
        )
        command.extend(["codex", "exec"])
        if request.session_id is not None:
            command.append("resume")
        command.extend(["-m", selection.model.model_id])
        settings = {
            "model_reasoning_effort": selection.effort,
        }
        if self.catalog_filename is not None:
            settings["model_catalog_json"] = str(
                PurePosixPath(home) / self.catalog_filename
            )
        if len(selection.model.service_tiers) > 1:
            settings["service_tier"] = selection.service_tier
        settings.update(
            {
                "mcp_servers.analyzer.command": self.mcp_python,
                "mcp_servers.analyzer.args": [self.mcp_server],
                "mcp_servers.analyzer.env.ANALYZER_MCP_SOURCE": self.environment.agent.analyzer_source,
                "mcp_servers.analyzer.env.ANALYZER_MCP_BASE_URL": self.environment.agent.analyzer_base_url,
                "mcp_servers.analyzer.env.VIBESIM_MANAGED_RUN_CONTEXT": self.environment.managed_context,
            }
        )
        for key, value in settings.items():
            command.extend(["-c", f"{key}={json.dumps(value, ensure_ascii=False)}"])
        command.extend(
            [
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "--json",
            ]
        )
        if request.structured_output:
            command.extend(["--output-schema", str(request.output_schema)])
        if request.session_id is not None:
            command.extend([request.session_id, "-"])
        return command
