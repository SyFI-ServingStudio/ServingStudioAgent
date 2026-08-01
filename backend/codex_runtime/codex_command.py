"""Build the Docker/Codex command line for one role call."""

from __future__ import annotations

import json

from ..managed_context import MANAGED_CONTEXT_CONTAINER_PATH
from .config import (
    ANALYZER_MCP_BASE_URL,
    ANALYZER_MCP_CONTAINER_DIR,
    ANALYZER_MCP_PYTHON,
    ANALYZER_MCP_SOURCE,
    CODEX_DOCKER_DG_USE_LOCAL_VERSION,
    CODEX_DOCKER_GID,
    CODEX_DOCKER_GPUS,
    CODEX_DOCKER_HOME,
    CODEX_DOCKER_UID,
    CODEX_DOCKER_USER,
    CODEX_DOCKER_UV_CACHE_DIR,
    CODEX_DOCKER_UV_PROJECT_ENVIRONMENT,
    MAIN_LOCK_SHA,
    role_codex_home_in_container,
)
from .exec_types import CodexExecRequest


def build_codex_exec_command(request: CodexExecRequest) -> list[str]:
    command = _docker_exec_prefix(request)
    command.extend(["codex", "exec"])

    codex_options = [
        # Model and effort are always explicit: both are per-conversation choices
        # now, so neither may fall back to the profile's config.toml default.
        "-m",
        request.model_id,
        "-c",
        f'model_reasoning_effort="{request.effort}"',
        "-c",
        f'mcp_servers.analyzer.command="{ANALYZER_MCP_PYTHON}"',
        "-c",
        f'mcp_servers.analyzer.args=["{ANALYZER_MCP_CONTAINER_DIR}/server.py"]',
        "-c",
        f"mcp_servers.analyzer.env.ANALYZER_MCP_SOURCE={json.dumps(ANALYZER_MCP_SOURCE)}",
        "-c",
        f"mcp_servers.analyzer.env.ANALYZER_MCP_BASE_URL={json.dumps(ANALYZER_MCP_BASE_URL)}",
        "-c",
        (
            "mcp_servers.analyzer.env.VIBESIM_MANAGED_RUN_CONTEXT="
            f"{json.dumps(MANAGED_CONTEXT_CONTAINER_PATH)}"
        ),
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
    ]
    # Codex constrains every text agent message in the current tool loop, not
    # only the final item, so fresh and resumed orchestrator calls must share
    # the exact same progress/milestone/terminal envelope contract.
    if request.output_schema:
        codex_options.extend(["--output-schema", request.output_schema])
    if request.is_resume:
        # `codex exec resume` only reads stdin when the prompt argument is "-".
        command.append("resume")
        command.extend(codex_options)
        command.extend([request.session_id or "", "-"])
        return command

    command.extend(codex_options)
    return command


def _docker_exec_prefix(request: CodexExecRequest) -> list[str]:
    return [
        "docker",
        "exec",
        "-i",
        "-u",
        f"{CODEX_DOCKER_UID}:{CODEX_DOCKER_GID}",
        "-e",
        f"HOME={CODEX_DOCKER_HOME}",
        "-e",
        f"CODEX_HOME={role_codex_home_in_container(request.label)}",
        "-e",
        f"USER={CODEX_DOCKER_USER}",
        "-e",
        f"LOGNAME={CODEX_DOCKER_USER}",
        "-e",
        f"UV_PROJECT_ENVIRONMENT={CODEX_DOCKER_UV_PROJECT_ENVIRONMENT}",
        "-e",
        f"UV_CACHE_DIR={CODEX_DOCKER_UV_CACHE_DIR}",
        "-e",
        f"VIBESIM_EXPECTED_LOCK_SHA={MAIN_LOCK_SHA}",
        "-e",
        f"DG_USE_LOCAL_VERSION={CODEX_DOCKER_DG_USE_LOCAL_VERSION}",
        "-e",
        f"CODEX_DOCKER_GPUS={CODEX_DOCKER_GPUS}",
        "-e",
        f"ANALYZER_MCP_SOURCE={ANALYZER_MCP_SOURCE}",
        "-e",
        f"ANALYZER_MCP_BASE_URL={ANALYZER_MCP_BASE_URL}",
        "-e",
        f"VIBESIM_MANAGED_RUN_CONTEXT={MANAGED_CONTEXT_CONTAINER_PATH}",
        "-e",
        f"VIBESIM_MANAGED_JOB_CONTEXT={MANAGED_CONTEXT_CONTAINER_PATH}",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-w",
        "/workspace",
        request.container,
    ]
