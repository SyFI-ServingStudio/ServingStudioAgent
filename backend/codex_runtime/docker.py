"""Docker container lifecycle and isolated Codex home setup."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from .commands import run_checked
from .config import (
    CODEX_DOCKER_AUTH_DIR,
    CODEX_DOCKER_DG_USE_LOCAL_VERSION,
    CODEX_DOCKER_GID,
    CODEX_DOCKER_GPUS,
    CODEX_DOCKER_HOME,
    CODEX_DOCKER_IMAGE,
    CODEX_DOCKER_UID,
    CODEX_DOCKER_USER,
    CODEX_DOCKER_UV_CACHE_DIR,
    CODEX_DOCKER_UV_PROJECT_ENVIRONMENT,
    CONTAINER_RUNTIME_VERSION,
    LOG,
    MAIN_DIR,
    MAIN_LOCK_SHA,
    WORKSPACES_DIR,
    codex_home_for,
    container_name,
)
from .workspace import main_submodule_paths
from ..logging_config import log_event


def _submodule_mount_args(conversation_id: str, container: str) -> list[str]:
    """Read-only bind-mount main's submodules (vLLM/TraceLab, ~5 GB) at their paths.

    The per-conversation workspace copy skips these gitlinks, so mounting the real
    checkout read-only makes their content available in-container without a ~5 GB
    per-conversation copy, while leaving the source tree untouched.
    """
    submodules = main_submodule_paths()
    mounts: list[str] = []
    for rel in submodules:
        src = MAIN_DIR / rel
        mounts.extend(["-v", f"{src}:/workspace/{rel.as_posix()}:ro"])
    if mounts:
        log_event(
            LOG,
            "container.submodule_mounts",
            conversation_id=conversation_id,
            container=container,
            paths=[str(p) for p in submodules],
        )
    return mounts


def remove_container(conversation_id: str) -> None:
    """Best-effort removal of the Docker container for a conversation/eval id."""
    container = container_name(conversation_id)
    log_event(LOG, "container.remove", conversation_id=conversation_id, container=container)
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)


def cleanup_conversation(conversation_id: str) -> None:
    """Best-effort cleanup for a deleted conversation."""
    remove_container(conversation_id)
    shutil.rmtree(WORKSPACES_DIR / conversation_id, ignore_errors=True)


def _copy_codex_auth_entry(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst)

def _prepare_codex_home(conversation_id: str) -> Path:
    """Create a clean per-conversation Codex home seeded with host auth.

    Mounting the host ``~/.codex`` directly leaks stale ``tmp`` / ``sessions`` /
    state DB paths into the Docker runtime. Copy only authentication/configuration
    inputs and let this isolated home own its runtime state.
    """
    host_codex_home = Path.home() / ".codex"
    if not host_codex_home.exists():
        raise RuntimeError(f"Codex auth directory not found: {host_codex_home}")

    codex_home = codex_home_for(conversation_id)
    codex_home.mkdir(parents=True, exist_ok=True)
    for name in (
        "auth.json",
        "config.toml",
        "installation_id",
        "version.json",
        "models_cache.json",
        ".personality_migration",
        "rules",
    ):
        _copy_codex_auth_entry(host_codex_home / name, codex_home / name)

    for runtime_dir in ("sessions", "tmp", "shell_snapshots", "log", "cache"):
        (codex_home / runtime_dir).mkdir(parents=True, exist_ok=True)
    return codex_home

def _docker_init_script() -> str:
    return f"""
set -euo pipefail
APP_UID={CODEX_DOCKER_UID}
APP_GID={CODEX_DOCKER_GID}
APP_USER={shlex.quote(CODEX_DOCKER_USER)}
APP_HOME={shlex.quote(CODEX_DOCKER_HOME)}
RUNTIME_VERSION={shlex.quote(CONTAINER_RUNTIME_VERSION)}
RUNTIME_IMAGE={shlex.quote(CODEX_DOCKER_IMAGE)}
EXPECTED_DG_USE_LOCAL_VERSION={shlex.quote(CODEX_DOCKER_DG_USE_LOCAL_VERSION)}
EXPECTED_LOCK_SHA={shlex.quote(MAIN_LOCK_SHA)}
GPU_REQUEST={shlex.quote(CODEX_DOCKER_GPUS)}

if [ "$(id -u)" != "$APP_UID" ] || [ "$(id -g)" != "$APP_GID" ]; then
  echo "Docker runner is not using the requested UID/GID: got $(id -u):$(id -g), expected $APP_UID:$APP_GID" >&2
  exit 126
fi

if [ ! -d "$APP_HOME" ] || [ ! -w "$APP_HOME" ]; then
  echo "prebuilt Docker image '$RUNTIME_IMAGE' does not provide writable home $APP_HOME for $APP_UID:$APP_GID" >&2
  echo "rebuild it with: user-facing-ui/scripts/build-codex-runner-image.sh" >&2
  exit 127
fi

mkdir -p "$APP_HOME" "$APP_HOME/.cache" "$APP_HOME/.local" "$APP_HOME/.npm"

if [ "${{DG_USE_LOCAL_VERSION:-}}" != "$EXPECTED_DG_USE_LOCAL_VERSION" ]; then
  echo "Docker runner has DG_USE_LOCAL_VERSION=${{DG_USE_LOCAL_VERSION:-unset}}, expected $EXPECTED_DG_USE_LOCAL_VERSION" >&2
  exit 127
fi

if [ "${{VIBESIM_BAKED_LOCK_SHA:-}}" != "$EXPECTED_LOCK_SHA" ]; then
  echo "Docker runner was prewarmed for lock ${{VIBESIM_BAKED_LOCK_SHA:-unset}}, expected $EXPECTED_LOCK_SHA" >&2
  exit 127
fi

if [ ! -d "${{UV_PROJECT_ENVIRONMENT:-}}" ] || [ ! -w "${{UV_PROJECT_ENVIRONMENT:-}}" ]; then
  echo "Docker runner does not provide writable baked uv env at ${{UV_PROJECT_ENVIRONMENT:-unset}}" >&2
  exit 127
fi

for tool in bash cargo git just node npm nvcc python python3 rustc uv codex; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "prebuilt Docker image '$RUNTIME_IMAGE' is missing required tool: $tool" >&2
    echo "build it with: user-facing-ui/scripts/build-codex-runner-image.sh" >&2
    exit 127
  fi
done

if [ "$GPU_REQUEST" != "" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Docker runner requested GPUs but nvidia-smi is not visible in the container" >&2
    exit 127
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "Docker runner requested GPUs but no NVIDIA GPU is visible in the container" >&2
    exit 127
  fi
fi

echo "$RUNTIME_VERSION" > /tmp/vibesim_ui_runtime_version
echo "$RUNTIME_IMAGE" > /tmp/vibesim_ui_runtime_image
echo "$GPU_REQUEST" > /tmp/vibesim_ui_gpu_request
echo "$EXPECTED_LOCK_SHA" > /tmp/vibesim_ui_main_lock_sha
touch /tmp/vibesim_ui_codex_ready
"""

def container_running(container: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"

def ensure_container(conversation_id: str, workspace_main: Path, mode: str) -> str:
    container = container_name(conversation_id)
    log_event(
        LOG,
        "container.ensure.start",
        conversation_id=conversation_id,
        container=container,
        workspace=str(workspace_main),
        mode=mode,
        uid=CODEX_DOCKER_UID,
        gid=CODEX_DOCKER_GID,
        docker_home=CODEX_DOCKER_HOME,
        runtime_version=CONTAINER_RUNTIME_VERSION,
        image=CODEX_DOCKER_IMAGE,
        main_lock_sha=MAIN_LOCK_SHA,
    )
    if container_running(container):
        gpu_ready_clause = (
            "&& command -v nvidia-smi >/dev/null 2>&1 "
            "&& nvidia-smi -L >/dev/null 2>&1 "
            if CODEX_DOCKER_GPUS
            else ""
        )
        ready = subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                f"DG_USE_LOCAL_VERSION={CODEX_DOCKER_DG_USE_LOCAL_VERSION}",
                "-e",
                f"CODEX_DOCKER_GPUS={CODEX_DOCKER_GPUS}",
                "-u",
                f"{CODEX_DOCKER_UID}:{CODEX_DOCKER_GID}",
                container,
                "bash",
                "-lc",
                (
                    "test -f /tmp/vibesim_ui_codex_ready "
                    f"&& test \"$(cat /tmp/vibesim_ui_runtime_version 2>/dev/null)\" = {CONTAINER_RUNTIME_VERSION!r} "
                    f"&& test \"$(cat /tmp/vibesim_ui_runtime_image 2>/dev/null)\" = {CODEX_DOCKER_IMAGE!r} "
                    f"&& test \"$(cat /tmp/vibesim_ui_gpu_request 2>/dev/null)\" = {CODEX_DOCKER_GPUS!r} "
                    f"&& test \"$(cat /tmp/vibesim_ui_main_lock_sha 2>/dev/null)\" = {MAIN_LOCK_SHA!r} "
                    f"&& test \"${{DG_USE_LOCAL_VERSION:-}}\" = {CODEX_DOCKER_DG_USE_LOCAL_VERSION!r} "
                    f"&& test \"${{VIBESIM_BAKED_LOCK_SHA:-}}\" = {MAIN_LOCK_SHA!r} "
                    f"&& test -d {CODEX_DOCKER_UV_PROJECT_ENVIRONMENT!r} "
                    f"&& test -w {CODEX_DOCKER_UV_PROJECT_ENVIRONMENT!r} "
                    "&& command -v bash >/dev/null 2>&1 "
                    "&& command -v cargo >/dev/null 2>&1 "
                    "&& command -v git >/dev/null 2>&1 "
                    "&& command -v just >/dev/null 2>&1 "
                    "&& command -v node >/dev/null 2>&1 "
                    "&& command -v npm >/dev/null 2>&1 "
                    "&& command -v nvcc >/dev/null 2>&1 "
                    "&& command -v python >/dev/null 2>&1 "
                    "&& command -v python3 >/dev/null 2>&1 "
                    "&& command -v rustc >/dev/null 2>&1 "
                    "&& command -v uv >/dev/null 2>&1 "
                    "&& command -v codex >/dev/null 2>&1 "
                    f"&& test -d {CODEX_DOCKER_AUTH_DIR!r} "
                    f"{gpu_ready_clause}"
                ),
            ],
            capture_output=True,
            check=False,
        )
        if ready.returncode == 0:
            log_event(
                LOG,
                "container.ensure.reuse",
                conversation_id=conversation_id,
                container=container,
            )
            return container
        log_event(
            LOG,
            "container.ensure.recreate_existing",
            conversation_id=conversation_id,
            container=container,
        )
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)

    log_event(LOG, "container.ensure.recreate", conversation_id=conversation_id, container=container)
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)

    auth_dir = _prepare_codex_home(conversation_id)
    log_event(
        LOG,
        "container.codex_home.ready",
        conversation_id=conversation_id,
        container=container,
        codex_home=str(auth_dir),
    )

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "--user",
        f"{CODEX_DOCKER_UID}:{CODEX_DOCKER_GID}",
        "-v",
        f"{workspace_main}:/workspace",
        "-v",
        f"{auth_dir}:{CODEX_DOCKER_AUTH_DIR}",
        *_submodule_mount_args(conversation_id, container),
        "-w",
        "/workspace",
        "-e",
        "PYTHONUNBUFFERED=1",
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
        f"VIBESIM_EXPECTED_LOCK_SHA={MAIN_LOCK_SHA}",
        "-e",
        f"DG_USE_LOCAL_VERSION={CODEX_DOCKER_DG_USE_LOCAL_VERSION}",
        "-e",
        f"CODEX_DOCKER_GPUS={CODEX_DOCKER_GPUS}",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
    ]
    if CODEX_DOCKER_GPUS:
        cmd.extend(["--gpus", CODEX_DOCKER_GPUS])
    cmd.extend([CODEX_DOCKER_IMAGE, "sleep", "infinity"])
    run_checked(cmd, timeout=180)

    run_checked(
        [
            "docker",
            "exec",
            "-e",
            f"DG_USE_LOCAL_VERSION={CODEX_DOCKER_DG_USE_LOCAL_VERSION}",
            "-e",
            f"CODEX_DOCKER_GPUS={CODEX_DOCKER_GPUS}",
            "-e",
            f"VIBESIM_EXPECTED_LOCK_SHA={MAIN_LOCK_SHA}",
            "-u",
            f"{CODEX_DOCKER_UID}:{CODEX_DOCKER_GID}",
            container,
            "bash",
            "-lc",
            _docker_init_script(),
        ],
        timeout=120,
    )
    log_event(LOG, "container.ensure.ready", conversation_id=conversation_id, container=container)
    return container
