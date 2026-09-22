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
    permission_mode: str = "auto"
    permission_prompts: str = "none"
    allowed_tools: tuple[str, ...] = ("Bash(uv run *)",)
    # `user` rather than nothing. Skill discovery follows the setting sources,
    # and dropping all three left Claude with its built-ins only -- the
    # workspace's own library was linked into the role home and never read.
    # The user source is that isolated home (`CLAUDE_CONFIG_DIR`), not the
    # operator's `~/.claude`, so this reads no settings a turn did not create.
    setting_sources: tuple[str, ...] = ("user",)

    def __post_init__(self):
        if self.permission_mode not in {
            "acceptEdits",
            "auto",
            "bypassPermissions",
            "dontAsk",
            "plan",
        }:
            raise ValueError("unsupported Claude permission mode")
        if self.permission_prompts not in {"host", "none"}:
            raise ValueError("unsupported Claude permission prompt target")
        if any(source not in {"user", "project", "local"} for source in self.setting_sources):
            raise ValueError("unsupported Claude setting source")
        if any("," in tool or not tool for tool in self.allowed_tools):
            # The flag is comma-or-space separated, so an embedded comma would
            # silently split one pattern into two broader ones.
            raise ValueError("Claude allowed tools must not contain commas")

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
        return request.execution.command(
            self.arguments(request),
            environment={
                "CLAUDE_CONFIG_DIR": home,
                "DISABLE_AUTOUPDATER": "1",
                "VIBESIM_AGENT_PID_FILE": pid_file,
            },
            inherited=self.inherited_environment,
            pid_file=pid_file,
            label="vibesim-claude",
        )

    def arguments(self, request: AgentRequest) -> list[str]:
        mcp = {
            "mcpServers": {
                "analyzer": {
                    "command": request.execution.mcp_python,
                    "args": [request.execution.mcp_server],
                    "env": {
                        "ANALYZER_MCP_SOURCE": request.execution.analyzer_source,
                        "ANALYZER_MCP_BASE_URL": request.execution.analyzer_base_url,
                        "VIBESIM_MANAGED_RUN_CONTEXT": request.execution.managed_context,
                    },
                }
            }
        }
        command = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            request.selection.model.model_id,
            "--effort",
            request.selection.effort,
            # Retired `--dangerously-skip-permissions`. `--permission-prompts
            # none` is load-bearing, not decoration: `auto` falls back to a
            # manual prompt when its classifier cannot evaluate an action, and
            # under `-p` the default `host` target has nobody to ask.
            "--permission-mode",
            self.permission_mode,
            "--permission-prompts",
            self.permission_prompts,
            "--setting-sources",
            ",".join(self.setting_sources),
            # A mount target in a container and a path outside the tree on the
            # host; either way it is the role contract, not the repo's own file.
            "--append-system-prompt-file",
            request.execution.agent_prompt,
            "--strict-mcp-config",
            "--mcp-config",
            json.dumps(mcp),
        ]
        if self.allowed_tools:
            # The isolated home carries no settings of its own, so without this
            # the workspace's own hot path would reach the classifier on each
            # call. Read-only commands are already allowed by the CLI itself.
            command.extend(["--allowedTools", ",".join(self.allowed_tools)])
        schema = self.output_schema(request)
        if schema is not None:
            # The CLI rejects the dialect URI; retain it in the validation copy.
            schema.pop("$schema", None)
            command.extend(["--json-schema", json.dumps(schema)])
        if request.session_id:
            command.extend(["--resume", request.session_id])
        return command
