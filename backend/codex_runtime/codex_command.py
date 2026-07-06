"""Build the Docker/Codex command line for one role call."""

from __future__ import annotations

from .config import (
    CODEX_DOCKER_DG_USE_LOCAL_VERSION,
    CODEX_DOCKER_GID,
    CODEX_DOCKER_GPUS,
    CODEX_DOCKER_HOME,
    CODEX_DOCKER_UID,
    CODEX_DOCKER_USER,
    CODEX_DOCKER_UV_CACHE_DIR,
    CODEX_DOCKER_UV_PROJECT_ENVIRONMENT,
    CODEX_MODEL,
    MAIN_LOCK_SHA,
)
from .exec_types import CodexExecRequest


def build_codex_exec_command(request: CodexExecRequest) -> list[str]:
    command = _docker_exec_prefix(request.container)
    command.extend(["codex", "exec"])

    codex_options = [
        "-m",
        CODEX_MODEL,
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
    ]
    if request.is_resume:
        # `codex exec resume` only reads stdin when the prompt argument is "-".
        command.append("resume")
        command.extend(codex_options)
        command.extend([request.session_id or "", "-"])
        return command

    if request.output_schema:
        codex_options.extend(["--output-schema", request.output_schema])
    command.extend(codex_options)
    return command


def _docker_exec_prefix(container: str) -> list[str]:
    return [
        "docker",
        "exec",
        "-i",
        "-u",
        f"{CODEX_DOCKER_UID}:{CODEX_DOCKER_GID}",
        "-e",
        f"HOME={CODEX_DOCKER_HOME}",
        "-e",
        f"USER={CODEX_DOCKER_USER}",
        "-e",
        f"LOGNAME={CODEX_DOCKER_USER}",
        "-e",
        f"UV_PROJECT_ENVIRONMENT={CODEX_DOCKER_UV_PROJECT_ENVIRONMENT}",
        "-e",
        f"UV_CACHE_DIR={CODEX_DOCKER_UV_CACHE_DIR}",
        "-e",
        f"MLSIM_EXPECTED_LOCK_SHA={MAIN_LOCK_SHA}",
        "-e",
        f"DG_USE_LOCAL_VERSION={CODEX_DOCKER_DG_USE_LOCAL_VERSION}",
        "-e",
        f"CODEX_DOCKER_GPUS={CODEX_DOCKER_GPUS}",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-w",
        "/workspace",
        container,
    ]
