"""Docker-backed two-role Codex runner for the MLSim chat UI.

Each conversation gets an isolated copy of git-tracked files from ``../main``.
Codex runs as the host UID/GID inside a prebuilt Docker image with that copy
mounted read/write at ``/workspace`` and an isolated Codex home mounted at the
container user's ``~/.codex`` path.

The loop is intentionally small:

1. The orchestrator returns JSON that either asks/notifies the user or delegates
   a concrete task to the implementer.
2. The implementer runs in the same copied workspace and returns free-form text.
3. The implementer summary is explicitly handed back to the orchestrator, which
   can continue delegating or finish with a user-facing message.

Each role keeps its own Codex session id and resumes it on later turns. There is
no judge or profiler here. The human user remains the control loop, but a single
browser turn can contain several orchestrator/implementer handoffs.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import html
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .logging_config import compact_text, log_event

WORKSPACE = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parents[1]
MAIN_DIR = WORKSPACE / "main"
WORKSPACES_DIR = UI_DIR / "workspaces"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

CODEX_DOCKER_IMAGE = os.environ.get("CODEX_DOCKER_IMAGE", "mlsim-ui-codex-runner:latest")
CODEX_MODEL = os.environ.get("CODEX_MODEL", "gpt-5.3-codex-spark")
CODEX_CALL_TIMEOUT = float(os.environ.get("CODEX_TURN_TIMEOUT", "600"))
CODEX_DOCKER_GPUS = os.environ.get("CODEX_DOCKER_GPUS", "all").strip()
CODEX_DOCKER_UID = int(os.environ.get("CODEX_DOCKER_UID", str(os.getuid())))
CODEX_DOCKER_GID = int(os.environ.get("CODEX_DOCKER_GID", str(os.getgid())))
CODEX_DOCKER_USER = os.environ.get("CODEX_DOCKER_USER") or os.environ.get("USER") or "codex"
CODEX_DOCKER_HOME = os.environ.get("CODEX_DOCKER_HOME", f"/home/{CODEX_DOCKER_USER}")
CODEX_DOCKER_AUTH_DIR = f"{CODEX_DOCKER_HOME}/.codex"
CODEX_DOCKER_DG_USE_LOCAL_VERSION = os.environ.get("CODEX_DOCKER_DG_USE_LOCAL_VERSION", "0")
CODEX_DOCKER_UV_PROJECT_ENVIRONMENT = os.environ.get(
    "CODEX_DOCKER_UV_PROJECT_ENVIRONMENT",
    "/opt/mlsim-venv",
)
CODEX_DOCKER_UV_CACHE_DIR = os.environ.get("CODEX_DOCKER_UV_CACHE_DIR", "/opt/mlsim-uv-cache")
CONTAINER_RUNTIME_VERSION = os.environ.get("CODEX_RUNNER_IMAGE_VERSION", "prebuilt-codex-runner-v5")
ORCHESTRATOR_SCHEMA_IN_CONTAINER = "/workspace/.codex/orchestrator.schema.json"
LOG = logging.getLogger("mlsim_ui.codex_runner")

EXECUTION_MODES = ("read-only", "workspace-write", "danger-full-access")
SANDBOX_MODES = EXECUTION_MODES  # compatibility with the existing API/UI naming
DEFAULT_SANDBOX = "workspace-write"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MAIN_LOCK_SHA = os.environ.get("CODEX_MAIN_LOCK_SHA") or _sha256_file(MAIN_DIR / "uv.lock")


def prompt_fingerprint() -> str:
    """Hash role prompts/schema so stale Codex sessions are not resumed."""
    digest = hashlib.sha256()
    for name in ("AGENTS.md", "orchestrator.txt", "implementer.txt", "orchestrator.schema.json"):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((PROMPTS_DIR / name).read_bytes())
        digest.update(b"\0")
    digest.update(CODEX_MODEL.encode("utf-8"))
    return digest.hexdigest()[:16]


def workspace_main_for(conversation_id: str) -> Path:
    return WORKSPACES_DIR / conversation_id / "main"


def codex_home_for(conversation_id: str) -> Path:
    return WORKSPACES_DIR / conversation_id / "codex-home"


def cleanup_conversation(conversation_id: str) -> None:
    """Best-effort cleanup for a deleted conversation."""
    container = _container_name(conversation_id)
    log_event(LOG, "conversation.cleanup", conversation_id=conversation_id, container=container)
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)
    shutil.rmtree(WORKSPACES_DIR / conversation_id, ignore_errors=True)


def _container_name(conversation_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", conversation_id)[:48]
    return f"mlsim-ui-{safe}"


def _run_checked(cmd: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"{(result.stderr or result.stdout).strip()[-2000:]}"
        )
    return result


def _git_tracked_files() -> list[Path]:
    result = _run_checked(["git", "-C", str(MAIN_DIR), "ls-files", "-z"], timeout=60)
    return [Path(raw) for raw in result.stdout.split("\0") if raw]


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


def _copy_tracked_file(rel_path: Path, dst_root: Path) -> None:
    src = MAIN_DIR / rel_path
    dst = dst_root / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if src.is_symlink():
        target = src.resolve(strict=False)
        if target.exists():
            if target.is_dir():
                shutil.copytree(target, dst, symlinks=True)
            else:
                shutil.copy2(target, dst)
        else:
            os.symlink(os.readlink(src), dst)
    else:
        shutil.copy2(src, dst)


def _refresh_workspace_agent_files(workspace_main: Path) -> None:
    """Copy runtime role instructions into the copied workspace.

    ``main`` currently has no tracked ``AGENTS.md``. Keeping this file inside the
    copied workspace makes shell/tool behavior consistent across orchestrator and
    implementer sessions. ``.codex/skills`` points at the copied repo-local skill
    tree so Codex can discover skills through its native workspace convention.
    """
    log_event(LOG, "workspace.refresh_agent_files", workspace=str(workspace_main))
    shutil.copy2(PROMPTS_DIR / "AGENTS.md", workspace_main / "AGENTS.md")
    codex_dir = workspace_main / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROMPTS_DIR / "orchestrator.schema.json", codex_dir / "orchestrator.schema.json")
    skills_link = codex_dir / "skills"
    if skills_link.exists() or skills_link.is_symlink():
        if skills_link.is_dir() and not skills_link.is_symlink():
            shutil.rmtree(skills_link)
        else:
            skills_link.unlink()
    os.symlink("../skills", skills_link)


def _ensure_workspace_git(workspace_main: Path) -> None:
    """Make the copied workspace a local git repo for branch/commit hygiene."""
    if (workspace_main / ".git").exists():
        log_event(LOG, "workspace.git.exists", workspace=str(workspace_main))
        return

    log_event(LOG, "workspace.git.init", workspace=str(workspace_main))
    _run_checked(["git", "-C", str(workspace_main), "init"], timeout=60)
    _run_checked(["git", "-C", str(workspace_main), "checkout", "-B", "main"], timeout=60)
    _run_checked(["git", "-C", str(workspace_main), "config", "user.name", "MLSim UI"], timeout=60)
    _run_checked(
        ["git", "-C", str(workspace_main), "config", "user.email", "mlsim-ui@example.invalid"],
        timeout=60,
    )
    _run_checked(["git", "-C", str(workspace_main), "add", "-A"], timeout=60)
    _run_checked(
        ["git", "-C", str(workspace_main), "commit", "-m", "Initial MLSim workspace snapshot"],
        timeout=120,
    )


def prepare_workspace(conversation_id: str) -> Path:
    """Create the per-conversation copy of git-tracked ``main`` files."""
    workspace_main = workspace_main_for(conversation_id)
    if workspace_main.exists():
        log_event(
            LOG,
            "workspace.prepare.reuse",
            conversation_id=conversation_id,
            workspace=str(workspace_main),
        )
        _refresh_workspace_agent_files(workspace_main)
        _ensure_workspace_git(workspace_main)
        return workspace_main

    log_event(
        LOG,
        "workspace.prepare.create",
        conversation_id=conversation_id,
        workspace=str(workspace_main),
    )
    tmp_root = workspace_main.parent.with_name(workspace_main.parent.name + ".tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_main = tmp_root / "main"
    tmp_main.mkdir(parents=True, exist_ok=True)

    for rel_path in _git_tracked_files():
        _copy_tracked_file(rel_path, tmp_main)
    _refresh_workspace_agent_files(tmp_main)
    _ensure_workspace_git(tmp_main)

    workspace_main.parent.parent.mkdir(parents=True, exist_ok=True)
    if workspace_main.parent.exists():
        shutil.rmtree(workspace_main.parent)
    tmp_root.replace(workspace_main.parent)
    return workspace_main


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

if [ "${{MLSIM_BAKED_LOCK_SHA:-}}" != "$EXPECTED_LOCK_SHA" ]; then
  echo "Docker runner was prewarmed for lock ${{MLSIM_BAKED_LOCK_SHA:-unset}}, expected $EXPECTED_LOCK_SHA" >&2
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

echo "$RUNTIME_VERSION" > /tmp/mlsim_ui_runtime_version
echo "$RUNTIME_IMAGE" > /tmp/mlsim_ui_runtime_image
echo "$GPU_REQUEST" > /tmp/mlsim_ui_gpu_request
echo "$EXPECTED_LOCK_SHA" > /tmp/mlsim_ui_main_lock_sha
touch /tmp/mlsim_ui_codex_ready
"""


def _container_running(container: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _ensure_container(conversation_id: str, workspace_main: Path, mode: str) -> str:
    container = _container_name(conversation_id)
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
    if _container_running(container):
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
                    "test -f /tmp/mlsim_ui_codex_ready "
                    f"&& test \"$(cat /tmp/mlsim_ui_runtime_version 2>/dev/null)\" = {CONTAINER_RUNTIME_VERSION!r} "
                    f"&& test \"$(cat /tmp/mlsim_ui_runtime_image 2>/dev/null)\" = {CODEX_DOCKER_IMAGE!r} "
                    f"&& test \"$(cat /tmp/mlsim_ui_gpu_request 2>/dev/null)\" = {CODEX_DOCKER_GPUS!r} "
                    f"&& test \"$(cat /tmp/mlsim_ui_main_lock_sha 2>/dev/null)\" = {MAIN_LOCK_SHA!r} "
                    f"&& test \"${{DG_USE_LOCAL_VERSION:-}}\" = {CODEX_DOCKER_DG_USE_LOCAL_VERSION!r} "
                    f"&& test \"${{MLSIM_BAKED_LOCK_SHA:-}}\" = {MAIN_LOCK_SHA!r} "
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
        f"MLSIM_EXPECTED_LOCK_SHA={MAIN_LOCK_SHA}",
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
    _run_checked(cmd, timeout=180)

    _run_checked(
        [
            "docker",
            "exec",
            "-e",
            f"DG_USE_LOCAL_VERSION={CODEX_DOCKER_DG_USE_LOCAL_VERSION}",
            "-e",
            f"CODEX_DOCKER_GPUS={CODEX_DOCKER_GPUS}",
            "-e",
            f"MLSIM_EXPECTED_LOCK_SHA={MAIN_LOCK_SHA}",
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


def _describe_item(item: dict[str, Any]) -> str:
    itype = item.get("type", "item")
    if itype in ("command_execution", "command", "local_shell_call"):
        cmd = item.get("command") or item.get("cmd") or item.get("action") or ""
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        return f"$ {str(cmd)[:240]}"
    if itype == "reasoning":
        return "...thinking"
    if itype in ("file_change", "patch", "apply_patch"):
        return "editing files..."
    if itype == "mcp_tool_call":
        return f"tool: {item.get('tool') or item.get('name') or ''}"
    text = item.get("text") or item.get("summary") or itype
    return str(text)[:240]


def _assistant_message_from_payload(payload: dict[str, Any]) -> tuple[str, str] | None:
    """Extract assistant text plus Codex phase from known JSON event payloads."""
    if payload.get("type") == "agent_message":
        text = payload.get("message") or payload.get("text") or ""
        return str(text), str(payload.get("phase") or "")

    if payload.get("type") != "message" or payload.get("role") != "assistant":
        return None

    content = payload.get("content") or []
    parts: list[str] = []
    if isinstance(content, list):
        for entry in content:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if text is None:
                text = entry.get("output_text") or entry.get("input_text")
            if text is not None:
                parts.append(str(text))
    elif isinstance(content, str):
        parts.append(content)

    return "".join(parts), str(payload.get("phase") or "")


def _find_rollout_file(conversation_id: str, session_id: str) -> Path | None:
    sessions_dir = codex_home_for(conversation_id) / "sessions"
    if not sessions_dir.exists():
        return None
    candidates = []
    for path in sessions_dir.rglob(f"*{session_id}*.jsonl"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _scan_rollout_agent_messages(
    rollout_file: Path,
    offset: int,
) -> tuple[list[tuple[str, str]], int]:
    try:
        size = rollout_file.stat().st_size
    except OSError:
        return [], offset
    if offset > size:
        offset = 0

    messages: list[tuple[str, str]] = []
    try:
        with rollout_file.open("rb") as file:
            file.seek(offset)
            for raw_line in file:
                try:
                    event = json.loads(raw_line.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                if event.get("type") not in ("event_msg", "response_item"):
                    continue
                payload = event.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                assistant_message = _assistant_message_from_payload(payload)
                if assistant_message is not None:
                    messages.append(assistant_message)
            return messages, file.tell()
    except OSError:
        return [], offset


def _translate(ev: dict[str, Any]) -> list[dict[str, str]]:
    etype = ev.get("type")
    if etype == "thread.started" and ev.get("thread_id"):
        return [{"kind": "session", "session_id": str(ev["thread_id"])}]
    if etype in ("event_msg", "response_item"):
        payload = ev.get("payload") or {}
        if isinstance(payload, dict):
            assistant_message = _assistant_message_from_payload(payload)
            if assistant_message is not None:
                text, phase = assistant_message
                return [{"kind": "agent_text", "text": text, "phase": phase}]
    if etype in ("item.completed", "item.started"):
        item = ev.get("item") or {}
        if item.get("type") == "agent_message":
            if etype == "item.completed":
                return [
                    {
                        "kind": "agent_text",
                        "text": item.get("text", ""),
                        "phase": item.get("phase", ""),
                    }
                ]
            return []
        if etype == "item.completed":
            return [{"kind": "progress", "text": _describe_item(item)}]
    if etype == "error":
        return [{"kind": "progress", "text": "warning: " + str(ev.get("message") or ev.get("error") or "error")}]
    return []


def _codex_stderr_for_error(stderr_text: str, *, returncode: int | None, has_final_text: bool) -> str:
    """Drop known Codex CLI bookkeeping noise after successful calls."""
    if returncode not in (0, None) or not has_final_text:
        return stderr_text
    ignored_patterns = (
        "Reading prompt from stdin",
        "failed to record rollout items: thread",
    )
    kept_lines = [
        line
        for line in stderr_text.splitlines()
        if not any(pattern in line for pattern in ignored_patterns)
    ]
    return "\n".join(kept_lines).strip()


async def _run_codex(
    container: str,
    prompt: str,
    *,
    label: str,
    conversation_id: str,
    turn_id: str,
    session_id: str | None = None,
    output_schema: str | None = None,
) -> AsyncIterator[dict[str, str]]:
    cmd = [
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
        "codex",
        "exec",
    ]
    codex_options = [
        "-m",
        CODEX_MODEL,
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
    ]
    if session_id:
        # `codex exec resume` only reads stdin when the prompt argument is "-".
        cmd.append("resume")
        cmd.extend(codex_options)
        cmd.extend([session_id, "-"])
    else:
        if output_schema:
            codex_options.extend(["--output-schema", output_schema])
        cmd.extend(codex_options)
    schema_arg_used = bool(output_schema and not session_id)
    log_event(
        LOG,
        "codex.start",
        conversation_id=conversation_id,
        turn_id=turn_id,
        role=label,
        container=container,
        resume=bool(session_id),
        codex_session_id=session_id,
        output_schema=output_schema if schema_arg_used else "",
        output_schema_requested=bool(output_schema),
        output_schema_arg_used=schema_arg_used,
        uid=CODEX_DOCKER_UID,
        gid=CODEX_DOCKER_GID,
        docker_home=CODEX_DOCKER_HOME,
        prompt_len=len(prompt),
        prompt_preview=compact_text(prompt),
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert proc.stdin is not None
    proc.stdin.write(prompt.encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()

    stderr_chunks: list[str] = []

    async def _drain_stderr() -> None:
        assert proc.stderr is not None
        async for line in proc.stderr:
            stderr_chunks.append(line.decode("utf-8", "replace"))

    err_task = asyncio.create_task(_drain_stderr())
    final_text: str | None = None
    seen_intermediate_outputs: set[tuple[str, str]] = set()
    current_session_id = session_id
    rollout_file: Path | None = None
    rollout_offset = 0
    if current_session_id:
        rollout_file = _find_rollout_file(conversation_id, current_session_id)
        if rollout_file is not None:
            with contextlib.suppress(OSError):
                rollout_offset = rollout_file.stat().st_size
    loop = asyncio.get_event_loop()
    deadline = loop.time() + CODEX_CALL_TIMEOUT
    timed_out = False

    def _poll_rollout_intermediate_outputs() -> list[dict[str, str]]:
        nonlocal rollout_file, rollout_offset
        if not current_session_id:
            return []
        if rollout_file is None:
            rollout_file = _find_rollout_file(conversation_id, current_session_id)
            if rollout_file is None:
                return []
        messages, rollout_offset = _scan_rollout_agent_messages(rollout_file, rollout_offset)
        notes = []
        for text, phase in messages:
            if phase != "commentary":
                continue
            note_text = text.strip()
            if not note_text:
                continue
            note_key = (label, note_text)
            if note_key in seen_intermediate_outputs:
                continue
            seen_intermediate_outputs.add(note_key)
            log_event(
                LOG,
                "codex.intermediate_output",
                conversation_id=conversation_id,
                turn_id=turn_id,
                role=label,
                text=note_text,
                source="rollout",
            )
            notes.append({"kind": "intermediate_output", "role": label, "text": note_text})
        return notes

    def _translate_stdout_line(raw_line: bytes) -> list[dict[str, str]]:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line:
            return []
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return []
        return _translate(ev)

    async def _emit_translated(out: dict[str, str]) -> AsyncIterator[dict[str, str]]:
        nonlocal current_session_id, final_text
        if out["kind"] == "session":
            current_session_id = out["session_id"]
            log_event(
                LOG,
                "codex.session",
                conversation_id=conversation_id,
                turn_id=turn_id,
                role=label,
                codex_session_id=out["session_id"],
            )
            yield {
                "kind": "session",
                "role": label,
                "session_id": out["session_id"],
            }
        elif out["kind"] == "agent_text":
            text = out.get("text") or ""
            phase = out.get("phase") or ""
            if phase == "commentary":
                note_text = text.strip()
                if not note_text:
                    return
                note_key = (label, note_text)
                if note_key in seen_intermediate_outputs:
                    return
                seen_intermediate_outputs.add(note_key)
                log_event(
                    LOG,
                    "codex.intermediate_output",
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    role=label,
                    text=note_text,
                    source="stdout",
                )
                yield {"kind": "intermediate_output", "role": label, "text": note_text}
            elif text.strip():
                final_text = text.strip()
        else:
            progress_text = out.get("text", "")
            log_event(
                LOG,
                "codex.progress",
                conversation_id=conversation_id,
                turn_id=turn_id,
                role=label,
                text=progress_text,
            )
            yield {"kind": "progress", "text": f"{label}: {progress_text}"}

    try:
        assert proc.stdout is not None
        stdout_buffer = b""
        read_task = asyncio.create_task(proc.stdout.read(8192))
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                timed_out = True
                break
            done, _pending = await asyncio.wait(
                {read_task},
                timeout=min(1.0, remaining),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for note in _poll_rollout_intermediate_outputs():
                yield note
            if not done:
                continue
            raw = read_task.result()
            if not raw:
                break
            stdout_buffer += raw
            while b"\n" in stdout_buffer:
                raw_line, stdout_buffer = stdout_buffer.split(b"\n", 1)
                for out in _translate_stdout_line(raw_line):
                    async for emitted in _emit_translated(out):
                        yield emitted
            read_task = asyncio.create_task(proc.stdout.read(8192))

        if not read_task.done():
            read_task.cancel()
            with contextlib.suppress(BaseException):
                await read_task
        if stdout_buffer.strip():
            for out in _translate_stdout_line(stdout_buffer):
                async for emitted in _emit_translated(out):
                    yield emitted
        for note in _poll_rollout_intermediate_outputs():
            yield note
        if timed_out:
            proc.kill()
        await proc.wait()
    except asyncio.CancelledError:
        log_event(
            LOG,
            "codex.cancelled",
            conversation_id=conversation_id,
            turn_id=turn_id,
            role=label,
            pid=proc.pid,
        )
        if proc.returncode is None:
            proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=5)
        raise
    finally:
        if not err_task.done():
            err_task.cancel()
        with contextlib.suppress(BaseException):
            await err_task

    if timed_out:
        log_event(
            LOG,
            "codex.timeout",
            conversation_id=conversation_id,
            turn_id=turn_id,
            role=label,
            timeout_s=CODEX_CALL_TIMEOUT,
        )
        yield {"kind": "error", "text": f"{label}: codex call timed out after {CODEX_CALL_TIMEOUT:.0f}s"}

    raw_stderr_text = "".join(stderr_chunks).strip()
    stderr_text = _codex_stderr_for_error(
        raw_stderr_text,
        returncode=proc.returncode,
        has_final_text=final_text is not None,
    )
    if final_text is None:
        if proc.returncode not in (0, None) and stderr_text:
            final_text = f"({label} exited {proc.returncode})\n\n```\n{stderr_text[-1500:]}\n```"
        elif stderr_text:
            final_text = f"({label} produced no final text)\n\n```\n{stderr_text[-1500:]}\n```"
        else:
            final_text = f"({label} produced no final text)"

    log_event(
        LOG,
        "codex.final",
        conversation_id=conversation_id,
        turn_id=turn_id,
        role=label,
        returncode=proc.returncode,
        final_len=len(final_text),
        final_preview=compact_text(final_text),
        stderr_len=len(stderr_text),
        stderr_tail=stderr_text[-500:] if stderr_text else "",
        raw_stderr_len=len(raw_stderr_text),
    )
    yield {"kind": "final", "text": final_text}


def _role_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _orchestrator_prompt(user_text: str, *, is_resume: bool) -> str:
    if is_resume:
        return user_text
    return f"{_role_prompt('orchestrator.txt')}\n\nNewest user message:\n{user_text}\n"


def _orchestrator_handoff_prompt(task: str, implementer_text: str) -> str:
    return (
        "The implementer returned a summary for your delegated task.\n\n"
        "You do not share the implementer Codex session. Treat the text below as "
        "the explicit handoff record, review it against your own orchestration "
        "context, and return exactly one JSON object.\n\n"
        "If the work is complete, risky, blocked, or needs a user choice, use "
        "`user_message`. If another bounded code-change, validation, or large "
        "exploration task is still needed, use `run_implementer` with that "
        "specific follow-up task.\n\n"
        "Delegated task:\n"
        f"{task}\n\n"
        "Implementer summary:\n"
        f"{implementer_text}\n"
    )


def _implementer_prompt(task: str, *, is_resume: bool) -> str:
    if is_resume:
        return f"Task:\n{task}\n"
    return f"{_role_prompt('implementer.txt')}\n\nTask:\n{task}\n"


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped] if stripped else []
    candidates.extend(m.strip() for m in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    return candidates


def _normalize_orchestrator_text_field(value: str) -> str:
    """Make text fields readable when the model double-escapes JSON newlines."""
    return (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
    )


def _parse_orchestrator(text: str) -> dict[str, Any] | None:
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        action = payload.get("action")
        if action == "run_implementer" and isinstance(payload.get("task"), str):
            return {
                "action": action,
                "task": _normalize_orchestrator_text_field(payload["task"]),
            }
        if action == "user_message" and isinstance(payload.get("message"), str):
            return {
                "action": action,
                "message": _normalize_orchestrator_text_field(payload["message"]),
            }
        if action is None and isinstance(payload.get("message"), str):
            return {
                "action": "user_message",
                "message": _normalize_orchestrator_text_field(payload["message"]),
            }
        if action is None and payload:
            return {
                "action": "user_message",
                "message": json.dumps(payload, ensure_ascii=False, indent=2),
            }
    return None


def _orchestrator_block(orchestrator_text: str) -> str:
    stripped = orchestrator_text.strip()
    parsed = _parse_orchestrator(stripped)
    if parsed is not None:
        body = json.dumps(parsed, ensure_ascii=False, indent=2)
        escaped_body = html.escape(body)
        return (
            '<details class="role-output orchestrator">\n'
            '<summary class="role-title">Orchestrator Raw</summary>\n'
            f'<pre class="role-raw"><code>{escaped_body}</code></pre>\n'
            "</details>"
        )
    escaped_text = html.escape(stripped)
    return (
        '<details class="role-output orchestrator">\n'
        '<summary class="role-title">Orchestrator Raw</summary>\n'
        f'<pre class="role-raw"><code>{escaped_text}</code></pre>\n'
        "</details>"
    )


def _compose_orchestrator_message(orchestrator_text: str, title: str, body: str) -> str:
    return f"{_orchestrator_block(orchestrator_text)}\n\n### {title}\n\n{body.strip()}"


def _format_implementer_summaries(summaries: list[str]) -> str:
    if not summaries:
        return ""
    if len(summaries) == 1:
        return summaries[0].strip()
    parts = []
    for idx, summary in enumerate(summaries, start=1):
        parts.append(f"**Round {idx}**\n\n{summary.strip()}")
    return "\n\n".join(parts)


def _compose_final_message(
    orchestrator_text: str,
    message: str,
    implementer_summaries: list[str],
) -> str:
    sections = [_orchestrator_block(orchestrator_text)]
    if implementer_summaries:
        sections.append(
            "### Implementer Summary\n\n"
            f"{_format_implementer_summaries(implementer_summaries)}"
        )
    sections.append(f"### Message\n\n{message.strip()}")
    return "\n\n".join(sections)


async def _prepare_runtime(conversation_id: str, mode: str) -> tuple[Path, str]:
    workspace_main = await asyncio.to_thread(prepare_workspace, conversation_id)
    container = await asyncio.to_thread(_ensure_container, conversation_id, workspace_main, mode)
    return workspace_main, container


async def run_turn(
    conversation_id: str,
    prompt: str,
    *,
    sandbox: str,
    sessions: dict[str, str] | None = None,
    turn_id: str = "",
    prompt_fingerprint: str = "",
) -> AsyncIterator[dict[str, str]]:
    """Run one user turn through orchestrator/implementer handoffs."""
    turn_id = turn_id or "unknown"
    mode = sandbox if sandbox in EXECUTION_MODES else DEFAULT_SANDBOX
    sessions = sessions or {}
    log_event(
        LOG,
        "turn.run.start",
        conversation_id=conversation_id,
        turn_id=turn_id,
        mode=mode,
        prompt_fingerprint=prompt_fingerprint,
        session_roles=sorted(sessions),
    )

    workspace_main_path = workspace_main_for(conversation_id)
    if workspace_main_path.exists():
        yield {"kind": "progress", "text": "checking isolated MLSim workspace..."}
    else:
        yield {"kind": "progress", "text": "creating isolated MLSim workspace..."}
    loop = asyncio.get_event_loop()
    workspace_started = loop.time()
    workspace_task = asyncio.create_task(asyncio.to_thread(prepare_workspace, conversation_id))
    while True:
        done, _pending = await asyncio.wait({workspace_task}, timeout=5)
        if done:
            _workspace_main = await workspace_task
            break
        elapsed = loop.time() - workspace_started
        yield {"kind": "progress", "text": f"workspace check still running ({elapsed:.0f}s)..."}

    container_name = _container_name(conversation_id)
    if await asyncio.to_thread(_container_running, container_name):
        yield {"kind": "progress", "text": "checking Docker Codex container..."}
    else:
        yield {"kind": "progress", "text": "starting Docker Codex container..."}
    container_started = loop.time()
    container_task = asyncio.create_task(asyncio.to_thread(_ensure_container, conversation_id, _workspace_main, mode))
    while True:
        done, _pending = await asyncio.wait({container_task}, timeout=10)
        if done:
            container = await container_task
            break
        elapsed = loop.time() - container_started
        yield {
            "kind": "progress",
            "text": f"Docker Codex container check still running ({elapsed:.0f}s)...",
        }

    log_event(
        LOG,
        "turn.runtime.ready",
        conversation_id=conversation_id,
        turn_id=turn_id,
        workspace=str(_workspace_main),
        container=container,
    )

    orchestrator_session = sessions.get("orchestrator")
    implementer_session = sessions.get("implementer")
    implementer_summaries: list[str] = []
    next_orchestrator_prompt = _orchestrator_prompt(prompt, is_resume=bool(orchestrator_session))

    while True:
        orchestrator_text: str | None = None
        async for ev in _run_codex(
            container,
            next_orchestrator_prompt,
            label="orchestrator",
            conversation_id=conversation_id,
            turn_id=turn_id,
            session_id=orchestrator_session,
            output_schema=ORCHESTRATOR_SCHEMA_IN_CONTAINER,
        ):
            if ev["kind"] == "session":
                orchestrator_session = ev["session_id"]
                yield ev
            elif ev["kind"] == "final":
                orchestrator_text = ev["text"]
            else:
                yield ev
        orchestrator_text = orchestrator_text or ""
        yield {"kind": "orchestrator", "text": orchestrator_text}

        decision = _parse_orchestrator(orchestrator_text)
        if decision is None:
            log_event(
                LOG,
                "orchestrator.parse_failed",
                conversation_id=conversation_id,
                turn_id=turn_id,
                text_preview=compact_text(orchestrator_text),
            )
            sections = [
                _orchestrator_block(orchestrator_text),
            ]
            if implementer_summaries:
                sections.append(
                    "### Implementer Summary\n\n"
                    f"{_format_implementer_summaries(implementer_summaries)}"
                )
            sections.append("### Error\n\nI could not parse the orchestrator decision.")
            yield {"kind": "final", "text": "\n\n".join(sections)}
            return

        log_event(
            LOG,
            "orchestrator.decision",
            conversation_id=conversation_id,
            turn_id=turn_id,
            action=decision["action"],
            message_len=len(decision.get("message", "")),
            task_len=len(decision.get("task", "")),
            preview=compact_text(decision.get("message") or decision.get("task") or ""),
        )
        if decision["action"] == "user_message":
            yield {
                "kind": "final",
                "text": _compose_final_message(
                    orchestrator_text,
                    decision["message"],
                    implementer_summaries,
                ),
            }
            return

        if mode == "read-only":
            body = (
                "This needs an implementer run, but this turn is in read-only mode. "
                "Switch the execution mode to workspace-write if you want me to run it "
                "inside the copied Docker workspace."
            )
            yield {
                "kind": "final",
                "text": _compose_final_message(orchestrator_text, body, implementer_summaries),
            }
            return

        task = decision["task"]
        yield {"kind": "progress", "text": "implementer: starting delegated task..."}
        implementer_text: str | None = None
        async for ev in _run_codex(
            container,
            _implementer_prompt(task, is_resume=bool(implementer_session)),
            label="implementer",
            conversation_id=conversation_id,
            turn_id=turn_id,
            session_id=implementer_session,
        ):
            if ev["kind"] == "session":
                implementer_session = ev["session_id"]
                yield ev
            elif ev["kind"] == "final":
                implementer_text = ev["text"]
                yield {"kind": "implementer", "text": implementer_text}
            else:
                yield ev

        if implementer_text is None:
            implementer_text = "(implementer produced no summary)"
        implementer_summaries.append(implementer_text)
        handoff_prompt = _orchestrator_handoff_prompt(task, implementer_text)
        if orchestrator_session:
            next_orchestrator_prompt = handoff_prompt
        else:
            next_orchestrator_prompt = f"{_role_prompt('orchestrator.txt')}\n\n{handoff_prompt}"
