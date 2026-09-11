"""Claude command construction from explicit environment and role contracts."""

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath

from ...runtime.command import ExecutionEnvironment
from ..base import AgentRequest


@dataclass(frozen=True)
class ClaudeCommand:
    environment: ExecutionEnvironment
    schemas: Mapping[str, dict]
    inherited_environment: tuple[str, ...] = ()
    mcp_python: str = "/opt/vibesim-analyzer-mcp-venv/bin/python"
    mcp_server: str = "/opt/vibesim/analyzer-evidence-mcp/server.py"

    def output_schema(self, request: AgentRequest) -> dict | None:
        if not request.structured_output:
            return None
        name = PurePosixPath(request.output_schema).name
        if (
            name
            not in {
                "assistant.schema.json",
                "orchestrator.schema.json",
                "implementer.schema.json",
            }
            or name not in self.schemas
        ):
            raise ValueError("unsupported agent output schema")
        return deepcopy(self.schemas[name])

    def build_tracked(
        self, request: AgentRequest, *, home: str, pid_file: str
    ) -> list[str]:
        mcp = {
            "mcpServers": {
                "analyzer": {
                    "command": self.mcp_python,
                    "args": [self.mcp_server],
                    "env": {
                        "ANALYZER_MCP_SOURCE": self.environment.agent.analyzer_source,
                        "ANALYZER_MCP_BASE_URL": self.environment.agent.analyzer_base_url,
                        "VIBESIM_MANAGED_RUN_CONTEXT": self.environment.managed_context,
                    },
                }
            }
        }
        command = self.environment.prefix(
            request.container,
            environment={
                "CLAUDE_CONFIG_DIR": home,
                "DISABLE_AUTOUPDATER": "1",
                "VIBESIM_AGENT_PID_FILE": pid_file,
            },
            inherited=self.inherited_environment,
        )
        command.extend(
            [
                "sh",
                "-c",
                (
                    'printf "%s\\n" "$$" > "$1" || exit; '
                    'if [ -e "$1.cancel" ]; then rm -f -- "$1"; exit 130; fi; '
                    'shift; exec "$@"'
                ),
                "vibesim-claude",
                pid_file,
                "claude",
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--model",
                request.selection.model.model_id,
                "--effort",
                request.selection.effort,
                "--dangerously-skip-permissions",
                "--setting-sources",
                "",
                "--append-system-prompt-file",
                "/workspace/AGENTS.md",
                "--strict-mcp-config",
                "--mcp-config",
                json.dumps(mcp),
            ]
        )
        schema = self.output_schema(request)
        if schema is not None:
            # The CLI rejects the dialect URI; retain it in the validation copy.
            schema.pop("$schema", None)
            command.extend(["--json-schema", json.dumps(schema)])
        if request.session_id:
            command.extend(["--resume", request.session_id])
        return command
