"""Explicit Claude CLI configuration for a Docker-isolated role."""

import json
from pathlib import PurePosixPath

from ..managed_context import MANAGED_CONTEXT_CONTAINER_PATH
from .codex_command import docker_exec_prefix
from .config import (
    ANALYZER_MCP_BASE_URL,
    ANALYZER_MCP_CONTAINER_DIR,
    ANALYZER_MCP_PYTHON,
    ANALYZER_MCP_SOURCE,
    CLAUDE_ENVIRONMENT,
    PROMPTS_DIR,
    role_codex_home_in_container,
)
from .exec_types import CodexExecRequest


def output_schema(request: CodexExecRequest) -> dict | None:
    if not request.output_schema:
        return None
    name = PurePosixPath(request.output_schema).name
    if name not in {
        "assistant.schema.json",
        "orchestrator.schema.json",
        "implementer.schema.json",
    }:
        raise ValueError("unsupported agent output schema")
    return json.loads((PROMPTS_DIR / name).read_text("utf-8"))


def pid_file(request: CodexExecRequest) -> str:
    return f"{role_codex_home_in_container(request.label)}/claude/call-{request.execution_id}.pid"


def build_claude_exec_command(request: CodexExecRequest) -> list[str]:
    mcp = {
        "mcpServers": {
            "analyzer": {
                "command": ANALYZER_MCP_PYTHON,
                "args": [f"{ANALYZER_MCP_CONTAINER_DIR}/server.py"],
                "env": {
                    "ANALYZER_MCP_SOURCE": ANALYZER_MCP_SOURCE,
                    "ANALYZER_MCP_BASE_URL": ANALYZER_MCP_BASE_URL,
                    "VIBESIM_MANAGED_RUN_CONTEXT": MANAGED_CONTEXT_CONTAINER_PATH,
                },
            }
        }
    }
    command = docker_exec_prefix(
        request,
        runtime_environment={
            "CLAUDE_CONFIG_DIR": f"{role_codex_home_in_container(request.label)}/claude",
            "DISABLE_AUTOUPDATER": "1",
            "VIBESIM_AGENT_PID_FILE": pid_file(request),
        },
        inherited_environment=CLAUDE_ENVIRONMENT,
    )
    command.extend(
        [
            "sh",
            "-c",
            'printf "%s\\n" "$$" > "$VIBESIM_AGENT_PID_FILE" && exec claude "$@"',
            "vibesim-claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            request.model_id,
            "--effort",
            request.effort,
            # The existing runtime delegates permission policy to its role contract
            # and Docker mounts. Match that policy inside this container only.
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
    schema = output_schema(request)
    if schema is not None:
        # Claude 2.1.250 rejects the Draft 2020-12 meta-schema URI. Our role
        # contracts use common object/string keywords; omit only the dialect
        # declaration here and retain the original for backend validation.
        schema.pop("$schema", None)
        command.extend(["--json-schema", json.dumps(schema)])
    if request.session_id:
        command.extend(["--resume", request.session_id])
    return command
