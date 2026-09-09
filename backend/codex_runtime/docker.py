"""Docker container lifecycle and isolated Codex home setup."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from ..logging_config import log_event
from .commands import run_checked
from .config import (
    ANALYZER_MCP_BASE_URL,
    ANALYZER_MCP_CONTAINER_DIR,
    ANALYZER_MCP_DIR,
    ANALYZER_MCP_SOURCE,
    CLAUDE_ENVIRONMENT,
    CODEX_DOCKER_CODEX_ROOT,
    CODEX_DOCKER_DG_USE_LOCAL_VERSION,
    CODEX_DOCKER_GID,
    CODEX_DOCKER_GPUS,
    CODEX_DOCKER_HF_HOME,
    CODEX_DOCKER_HOME,
    CODEX_DOCKER_IMAGE,
    CODEX_DOCKER_UID,
    CODEX_DOCKER_USER,
    CODEX_DOCKER_UV_CACHE_DIR,
    CODEX_DOCKER_UV_PROJECT_ENVIRONMENT,
    CONTAINER_RUNTIME_VERSION,
    DEFAULT_AGENT_MODE,
    DEFAULT_CODEX_FAMILY,
    HOST_HF_HOME,
    LOG,
    MAIN_DIR,
    MAIN_LOCK_SHA,
    PROMPTS_CONTAINER_DIR,
    PROMPTS_DIR,
    agents_prompt_name,
    codex_family,
    codex_home_for,
    container_name,
    role_codex_home_for,
    role_codex_home_in_container,
    roles_for_agent_mode,
)
from .workspace import main_submodule_paths


def _submodule_mount_args(conversation_id: str, container: str) -> list[str]:
    """Read-only bind-mount main's submodules (vLLM/TraceLab, ~5 GB) at their paths.

    The managed workspace copy skips these gitlinks, so mounting the real
    checkout read-only makes their content available in-container without a ~5 GB
    copy per workspace, while leaving the source tree untouched.
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


def _candidate_mount_args(
    conversation_id: str, container: str, peer_dir: str | None
) -> list[str]:
    """Read-only bind-mount THIS conversation's co-evolution peer at /candidate.

    ``peer_dir`` is the vibe-serve candidate workspace that the caller which
    created this conversation asked to expose (persisted per-conversation at
    create time — see ``store.create(peer_workspace=...)``). Only the conversation
    that was created with a peer gets the mount; every other conversation is
    unaffected. A missing/absent directory -> no mount.
    """
    if not peer_dir:
        return []
    peer = Path(peer_dir).expanduser()
    if not peer.is_dir():
        log_event(
            LOG,
            "container.candidate_mount_skipped",
            conversation_id=conversation_id,
            container=container,
            path=str(peer),
        )
        return []
    log_event(
        LOG,
        "container.candidate_mount",
        conversation_id=conversation_id,
        container=container,
        path=str(peer),
    )
    return ["-v", f"{peer.resolve()}:/candidate:ro"]


def _model_mount_args(conversation_id: str, container: str) -> list[str]:
    """Expose the configured host Hugging Face cache read-only at ``/model``."""
    if HOST_HF_HOME is None:
        return []
    if not HOST_HF_HOME.is_dir():
        raise RuntimeError(f"configured HF_HOME directory not found: {HOST_HF_HOME}")
    resolved_hf_home = HOST_HF_HOME.resolve()
    log_event(
        LOG,
        "container.model_mount",
        conversation_id=conversation_id,
        container=container,
        path=str(resolved_hf_home),
        destination=CODEX_DOCKER_HF_HOME,
    )
    return [
        "-v",
        f"{resolved_hf_home}:{CODEX_DOCKER_HF_HOME}:ro",
        "-e",
        f"HF_HOME={CODEX_DOCKER_HF_HOME}",
    ]


def _agent_prompt_mount_args(
    workspace_main: Path,
    *,
    autonomous: bool,
    agent_mode: str,
) -> list[str]:
    """Mount this conversation's selected project instructions read-only.

    Every shared workspace must already contain a regular ``AGENTS.md`` target.
    Failing closed here is important: Docker's short ``-v`` syntax can create a
    missing bind target inside the host workspace, which would violate the
    runtime's no-mutation contract for ``w_main``.
    """
    prompt_path = PROMPTS_DIR / agents_prompt_name(autonomous, agent_mode)
    if not prompt_path.is_file():
        raise RuntimeError(f"agent prompt not found: {prompt_path}")
    mount_target = workspace_main / "AGENTS.md"
    if not mount_target.is_file():
        raise RuntimeError(
            "workspace AGENTS.md mount target is missing; "
            f"refusing to modify shared workspace: {mount_target}"
        )
    return [
        "--mount",
        (f"type=bind,src={prompt_path.resolve()},dst=/workspace/AGENTS.md,readonly"),
    ]


def remove_container(workspace_id: str, conversation_id: str) -> None:
    """Best-effort removal of the Docker container for a conversation/eval id."""
    container = container_name(workspace_id, conversation_id)
    log_event(
        LOG,
        "container.remove",
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        container=container,
    )
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)


def cleanup_conversation(workspace_id: str, conversation_id: str) -> None:
    """Remove ephemeral runtime state without touching the shared workspace."""
    remove_container(workspace_id, conversation_id)
    shutil.rmtree(codex_home_for(workspace_id, conversation_id), ignore_errors=True)


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


def _import_legacy_shared_runtime(legacy_home: Path, role_home: Path) -> None:
    """Copy pre-role-isolation session state into one role home once.

    Older conversations stored both role sessions directly below the
    conversation Codex home.  The database still identifies the correct role
    session, so copying the legacy trees into both isolated homes preserves
    resume behavior without moving or deleting user-owned history.
    """
    migration_marker = role_home / ".legacy-shared-runtime-imported"
    if migration_marker.exists():
        return

    for runtime_name in ("sessions", "shell_snapshots"):
        source = legacy_home / runtime_name
        if source.is_dir():
            shutil.copytree(
                source,
                role_home / runtime_name,
                symlinks=True,
                dirs_exist_ok=True,
            )

    migration_marker.write_text("role-isolation-v1\n", encoding="utf-8")


def _prepare_role_codex_home(
    workspace_id: str,
    conversation_id: str,
    role: str,
    family_id: str,
) -> Path:
    """Prepare one role's durable Codex home from its family's auth profile.

    Keyed by family, not model: sibling models share one auth home, which is what
    lets a rollout recorded by one of them resume under another. Runtime state is
    never copied from the source profile, and keeping role homes separate stops a
    session created against one provider from being resumed by the other role.
    """
    family = codex_family(family_id)
    if family.runner == "claude":
        if not family.available:
            raise RuntimeError(
                "Claude credentials are missing; configure ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or CLAUDE_CODE_OAUTH_TOKEN"
            )
        home = role_codex_home_for(workspace_id, conversation_id, role) / "claude"
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        skills = home / "skills"
        if not skills.is_symlink() and not skills.exists():
            skills.symlink_to("/workspace/skills", target_is_directory=True)
        return home
    host_codex_home = family.host_codex_home
    if not host_codex_home.exists():
        raise RuntimeError(
            f"Codex family {family_id!r} home not found: {host_codex_home}"
        )
    missing_environment = [
        name
        for name in family.required_environment
        if not os.environ.get(name, "").strip()
    ]
    if missing_environment:
        raise RuntimeError(
            f"Codex family {family_id!r} is missing environment: "
            + ", ".join(missing_environment)
        )

    legacy_home = codex_home_for(workspace_id, conversation_id)
    codex_home = role_codex_home_for(workspace_id, conversation_id, role)
    codex_home.mkdir(parents=True, exist_ok=True)
    _import_legacy_shared_runtime(legacy_home, codex_home)
    for name in (
        "auth.json",
        "config.toml",
        "installation_id",
        "version.json",
        "models_cache.json",
        "models_catalog.json",
        ".personality_migration",
        "rules",
    ):
        _copy_codex_auth_entry(host_codex_home / name, codex_home / name)

    config_path = codex_home / "config.toml"
    if config_path.is_file() and (codex_home / "models_catalog.json").is_file():
        config_text = config_path.read_text("utf-8")
        container_catalog = f"{role_codex_home_in_container(role)}/models_catalog.json"
        config_text = re.sub(
            r"^model_catalog_json\s*=.*$",
            f'model_catalog_json = "{container_catalog}"',
            config_text,
            flags=re.MULTILINE,
        )
        config_path.write_text(config_text, "utf-8")

    for runtime_dir in ("sessions", "tmp", "shell_snapshots", "log", "cache"):
        (codex_home / runtime_dir).mkdir(parents=True, exist_ok=True)
    return codex_home


def _docker_init_script(backend_fingerprint: str) -> str:
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
MODEL_HOME={shlex.quote(CODEX_DOCKER_HF_HOME)}
MODEL_MOUNT_REQUIRED={"1" if HOST_HF_HOME is not None else "0"}
BACKEND_FINGERPRINT={shlex.quote(backend_fingerprint)}

if [ "$(id -u)" != "$APP_UID" ] || [ "$(id -g)" != "$APP_GID" ]; then
  echo "Docker runner is not using the requested UID/GID: got $(id -u):$(id -g), expected $APP_UID:$APP_GID" >&2
  exit 126
fi

if [ ! -d "$APP_HOME" ] || [ ! -w "$APP_HOME" ]; then
  echo "prebuilt Docker image '$RUNTIME_IMAGE' does not provide writable home $APP_HOME for $APP_UID:$APP_GID" >&2
  echo "rebuild it with: VibeSimAgent/scripts/build-codex-runner-image.sh" >&2
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

if [ ! -d "${{VIBESIM_BAKED_TARGET:-}}/release" ]; then
  echo "prebuilt Docker image '$RUNTIME_IMAGE' is missing its Cargo target seed" >&2
  exit 127
fi

if [ ! -d /workspace/target ]; then
  mkdir -p /workspace/target
  cp -a --reflink=auto "${{VIBESIM_BAKED_TARGET}}/." /workspace/target/
fi

if [ ! -d "${{UV_PROJECT_ENVIRONMENT:-}}" ] || [ ! -w "${{UV_PROJECT_ENVIRONMENT:-}}" ]; then
  echo "Docker runner does not provide writable baked uv env at ${{UV_PROJECT_ENVIRONMENT:-unset}}" >&2
  exit 127
fi

for tool in bash cargo git just node npm nvcc python python3 rustc uv codex; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "prebuilt Docker image '$RUNTIME_IMAGE' is missing required tool: $tool" >&2
    echo "build it with: VibeSimAgent/scripts/build-codex-runner-image.sh" >&2
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

if [ "$MODEL_MOUNT_REQUIRED" = "1" ]; then
  if [ "${{HF_HOME:-}}" != "$MODEL_HOME" ] || [ ! -d "$MODEL_HOME" ] || [ ! -r "$MODEL_HOME" ]; then
    echo "Docker runner is missing the configured Hugging Face cache at $MODEL_HOME" >&2
    exit 127
  fi
fi

echo "$RUNTIME_VERSION" > /tmp/vibesim_ui_runtime_version
echo "$RUNTIME_IMAGE" > /tmp/vibesim_ui_runtime_image
echo "$GPU_REQUEST" > /tmp/vibesim_ui_gpu_request
echo "$EXPECTED_LOCK_SHA" > /tmp/vibesim_ui_main_lock_sha
echo "$BACKEND_FINGERPRINT" > /tmp/vibesim_ui_backend_fingerprint
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


def ensure_container(
    workspace_id: str,
    conversation_id: str,
    workspace_main: Path,
    mode: str,
    peer_dir: str | None = None,
    *,
    autonomous: bool = False,
    agent_mode: str = DEFAULT_AGENT_MODE,
    role_families: dict[str, str] | None = None,
) -> str:
    container = container_name(workspace_id, conversation_id)
    # Only the roles this agent mode runs get a Codex home, a fingerprint entry,
    # and a readiness check. Switching mode therefore changes the fingerprint and
    # recreates the container, which is what seeds the new role's home.
    family_selection = {
        role: (role_families or {}).get(role, DEFAULT_CODEX_FAMILY)
        for role in roles_for_agent_mode(agent_mode)
    }
    # Fingerprinted by family, not model: switching to a sibling model reuses the
    # same auth home, so tearing the container down would buy nothing. Role order
    # follows `roles_for_agent_mode`, so an orchestrated conversation keeps the
    # exact fingerprint string it had before single mode existed.
    backend_fingerprint = ",".join(
        f"{role}:{family_id}" for role, family_id in family_selection.items()
    )
    for role, family_id in family_selection.items():
        _prepare_role_codex_home(workspace_id, conversation_id, role, family_id)
    codex_root = codex_home_for(workspace_id, conversation_id)
    selected_agents_prompt = agents_prompt_name(autonomous, agent_mode)
    role_home_ready_clause = "".join(
        f"&& test -d {role_codex_home_in_container(role)!r} "
        for role in family_selection
    )
    claude_required = any(
        codex_family(family).runner == "claude" for family in family_selection.values()
    )
    if claude_required:
        role_home_ready_clause += "&& command -v claude >/dev/null 2>&1 "
    log_event(
        LOG,
        "container.ensure.start",
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        container=container,
        workspace=str(workspace_main),
        mode=mode,
        agents_prompt=selected_agents_prompt,
        uid=CODEX_DOCKER_UID,
        gid=CODEX_DOCKER_GID,
        docker_home=CODEX_DOCKER_HOME,
        runtime_version=CONTAINER_RUNTIME_VERSION,
        image=CODEX_DOCKER_IMAGE,
        main_lock_sha=MAIN_LOCK_SHA,
    )
    if container_running(container):
        gpu_ready_clause = (
            "&& command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 "
            if CODEX_DOCKER_GPUS
            else ""
        )
        model_ready_clause = (
            f'&& test "${{HF_HOME:-}}" = {CODEX_DOCKER_HF_HOME!r} '
            f"&& test -d {CODEX_DOCKER_HF_HOME!r} "
            f"&& test -r {CODEX_DOCKER_HF_HOME!r} "
            if HOST_HF_HOME is not None
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
                    f'&& test "$(cat /tmp/vibesim_ui_runtime_version 2>/dev/null)" = {CONTAINER_RUNTIME_VERSION!r} '
                    f'&& test "$(cat /tmp/vibesim_ui_runtime_image 2>/dev/null)" = {CODEX_DOCKER_IMAGE!r} '
                    f'&& test "$(cat /tmp/vibesim_ui_gpu_request 2>/dev/null)" = {CODEX_DOCKER_GPUS!r} '
                    f'&& test "$(cat /tmp/vibesim_ui_main_lock_sha 2>/dev/null)" = {MAIN_LOCK_SHA!r} '
                    f'&& test "$(cat /tmp/vibesim_ui_backend_fingerprint 2>/dev/null)" = {backend_fingerprint!r} '
                    f'&& test "${{DG_USE_LOCAL_VERSION:-}}" = {CODEX_DOCKER_DG_USE_LOCAL_VERSION!r} '
                    f'&& test "${{VIBESIM_BAKED_LOCK_SHA:-}}" = {MAIN_LOCK_SHA!r} '
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
                    f"{role_home_ready_clause}"
                    "&& test -r /workspace/AGENTS.md "
                    f"&& cmp -s /workspace/AGENTS.md "
                    f"{(PROMPTS_CONTAINER_DIR + '/' + selected_agents_prompt)!r} "
                    f"&& test -r {ANALYZER_MCP_CONTAINER_DIR + '/server.py'!r} "
                    f"{gpu_ready_clause}"
                    f"{model_ready_clause}"
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
        subprocess.run(
            ["docker", "rm", "-f", container], capture_output=True, check=False
        )

    log_event(
        LOG,
        "container.ensure.recreate",
        conversation_id=conversation_id,
        container=container,
    )
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)

    log_event(
        LOG,
        "container.codex_home.ready",
        conversation_id=conversation_id,
        container=container,
        codex_home=str(codex_root),
        families=family_selection,
    )

    cmd = [
        "docker",
        "run",
        "-d",
        "--init",
        "--name",
        container,
        "--add-host",
        "host.docker.internal:host-gateway",
        "--user",
        f"{CODEX_DOCKER_UID}:{CODEX_DOCKER_GID}",
        "-v",
        f"{workspace_main}:/workspace",
        "-v",
        f"{codex_root}:{CODEX_DOCKER_CODEX_ROOT}",
        "-v",
        f"{ANALYZER_MCP_DIR}:{ANALYZER_MCP_CONTAINER_DIR}:ro",
        "-v",
        f"{PROMPTS_DIR}:{PROMPTS_CONTAINER_DIR}:ro",
        *_agent_prompt_mount_args(
            workspace_main,
            autonomous=autonomous,
            agent_mode=agent_mode,
        ),
        *_submodule_mount_args(conversation_id, container),
        *_candidate_mount_args(conversation_id, container, peer_dir),
        *_model_mount_args(conversation_id, container),
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
        f"ANALYZER_MCP_SOURCE={ANALYZER_MCP_SOURCE}",
        "-e",
        f"ANALYZER_MCP_BASE_URL={ANALYZER_MCP_BASE_URL}",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
    ]
    required_environment = {
        environment_name
        for family_id in family_selection.values()
        for environment_name in codex_family(family_id).required_environment
    }
    for environment_name in sorted(required_environment):
        # Passing only the name keeps the secret value out of command logs;
        # Docker copies it from the backend process environment.
        cmd.extend(["-e", environment_name])
    if claude_required:
        for environment_name in CLAUDE_ENVIRONMENT:
            if os.environ.get(environment_name, "").strip():
                cmd.extend(["-e", environment_name])
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
            _docker_init_script(backend_fingerprint)
            + ("\ncommand -v claude >/dev/null\n" if claude_required else ""),
        ],
        timeout=300,
    )
    log_event(
        LOG,
        "container.ensure.ready",
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        container=container,
    )
    return container
