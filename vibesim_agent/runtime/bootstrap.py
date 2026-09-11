"""Prebuilt runner validation and seed copy, with no runtime compilation."""

import shlex

from .command import ExecutionEnvironment
from .mounts import HF_TARGET, MCP_TARGET, WORKSPACE_TARGET

READY_MARKER = "/tmp/vibesim_agent_ready"
BASE_BINARIES = (
    "bash",
    "cargo",
    "git",
    "just",
    "node",
    "npm",
    "nvcc",
    "python",
    "python3",
    "rustc",
    "uv",
)


def _preamble(
    environment: ExecutionEnvironment,
    *,
    fingerprint: str,
    binaries: tuple[str, ...],
    role_homes: tuple[str, ...],
    agent_prompt: str,
) -> str:
    settings = environment.container
    values = {
        "APP_UID": str(settings.uid),
        "APP_GID": str(settings.gid),
        "APP_USER": settings.user,
        "APP_HOME": str(settings.home),
        "UV_ENV": str(settings.uv_project_environment),
        "EXPECTED_DG_USE_LOCAL_VERSION": str(int(settings.dg_use_local_version)),
        "EXPECTED_LOCK_SHA": environment.lock_sha,
        "GPU_REQUEST": settings.gpus,
        "MODEL_HOME": str(HF_TARGET),
        "MODEL_MOUNT_REQUIRED": "1" if settings.hf_home is not None else "0",
        "EXPECTED_ANALYZER_SOURCE": environment.agent.analyzer_source,
        "EXPECTED_ANALYZER_URL": environment.agent.analyzer_base_url,
        "WORKSPACE": str(WORKSPACE_TARGET),
        "MCP_SERVER": str(MCP_TARGET / "server.py"),
        "AGENT_PROMPT": agent_prompt,
        "READY_MARKER": READY_MARKER,
        "FINGERPRINT": fingerprint,
    }
    assignments = [f"{name}={shlex.quote(value)}" for name, value in values.items()]
    assignments.append(
        "REQUIRED_TOOLS=("
        + " ".join(
            shlex.quote(tool) for tool in dict.fromkeys((*BASE_BINARIES, *binaries))
        )
        + ")"
    )
    assignments.append(
        "ROLE_HOMES=(" + " ".join(shlex.quote(home) for home in role_homes) + ")"
    )
    return (
        "set -euo pipefail\n"
        + "\n".join(assignments)
        + "\n"
        + r"""
fail() { printf '%s\n' "$1" >&2; exit "${2:-127}"; }
"""
    )


def _checks() -> str:
    return r"""
if [ "$(id -u)" != "$APP_UID" ] || [ "$(id -g)" != "$APP_GID" ]; then
  fail "Docker runner is not using the requested UID/GID" 126
fi
if [ "${HOME:-}" != "$APP_HOME" ] || [ "${USER:-}" != "$APP_USER" ] || [ "${LOGNAME:-}" != "$APP_USER" ]; then
  fail "Docker runner home/user environment does not match configuration"
fi
if [ ! -d "$APP_HOME" ] || [ ! -w "$APP_HOME" ]; then
  fail "prebuilt Docker image does not provide the configured writable home"
fi
if [ "${DG_USE_LOCAL_VERSION:-}" != "$EXPECTED_DG_USE_LOCAL_VERSION" ]; then
  fail "Docker runner DG_USE_LOCAL_VERSION does not match configuration"
fi
if [ "${VIBESIM_BAKED_LOCK_SHA:-}" != "$EXPECTED_LOCK_SHA" ]; then
  fail "Docker runner baked lock does not match the requested lock"
fi
if [ -z "${VIBESIM_BAKED_TARGET:-}" ] || [ ! -d "$VIBESIM_BAKED_TARGET/release" ]; then
  fail "prebuilt Docker image is missing its Cargo target seed"
fi
if [ "${UV_PROJECT_ENVIRONMENT:-}" != "$UV_ENV" ] || [ ! -d "$UV_ENV" ] || [ ! -w "$UV_ENV" ]; then
  fail "Docker runner does not provide the configured writable baked uv environment"
fi
for tool in "${REQUIRED_TOOLS[@]}"; do
  if ! command -v -- "$tool" >/dev/null 2>&1; then
    fail "prebuilt Docker image is missing required tool: $tool"
  fi
done
for role_home in "${ROLE_HOMES[@]}"; do
  if [ ! -d "$role_home" ] || [ ! -w "$role_home" ]; then
    fail "Docker runner is missing a writable role home"
  fi
done
if [ ! -r "$WORKSPACE/AGENTS.md" ] || [ ! -r "$AGENT_PROMPT" ] || ! cmp -s -- "$WORKSPACE/AGENTS.md" "$AGENT_PROMPT"; then
  fail "Docker runner project instructions do not match the selected prompt"
fi
if [ ! -r "$MCP_SERVER" ]; then
  fail "Docker runner is missing the Analyzer MCP server"
fi
if [ "${ANALYZER_MCP_SOURCE:-}" != "$EXPECTED_ANALYZER_SOURCE" ] || [ "${ANALYZER_MCP_BASE_URL:-}" != "$EXPECTED_ANALYZER_URL" ]; then
  fail "Docker runner Analyzer environment does not match configuration"
fi
if [ "$GPU_REQUEST" != "" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    fail "Docker runner requested GPUs but no NVIDIA GPU is visible"
  fi
fi
if [ "$MODEL_MOUNT_REQUIRED" = "1" ]; then
  if [ "${HF_HOME:-}" != "$MODEL_HOME" ] || [ ! -d "$MODEL_HOME" ] || [ ! -r "$MODEL_HOME" ]; then
    fail "Docker runner is missing the configured Hugging Face cache"
  fi
fi
"""


def bootstrap_script(
    environment: ExecutionEnvironment,
    *,
    fingerprint: str,
    binaries: tuple[str, ...],
    role_homes: tuple[str, ...],
    agent_prompt: str,
) -> str:
    """Invalidate readiness first, seed an absent target, then publish readiness."""
    return (
        _preamble(
            environment,
            fingerprint=fingerprint,
            binaries=binaries,
            role_homes=role_homes,
            agent_prompt=agent_prompt,
        )
        + r"""
rm -f -- "$READY_MARKER"
"""
        + _checks()
        + r"""
mkdir -p -- "$APP_HOME/.cache" "$APP_HOME/.local" "$APP_HOME/.npm"
if [ ! -d "$WORKSPACE/target" ]; then
  SEED_STAGE=$(mktemp -d "$WORKSPACE/.vibesim-target.XXXXXXXX")
  trap 'rm -rf -- "$SEED_STAGE"' EXIT
  cp -a --reflink=auto "$VIBESIM_BAKED_TARGET/." "$SEED_STAGE/"
  test -d "$SEED_STAGE/release" || fail "copied target seed is missing its release directory"
  # The complete seed becomes visible at once. A concurrent existing target wins.
  if ! mv -T -n -- "$SEED_STAGE" "$WORKSPACE/target"; then
    test -d "$WORKSPACE/target" || fail "could not publish copied target seed"
  fi
  rm -rf -- "$SEED_STAGE"
  trap - EXIT
fi
test -d "$WORKSPACE/target" || fail "workspace target is missing"
printf '%s\n' "$FINGERPRINT" > "$READY_MARKER"
"""
    )


def readiness_script(
    environment: ExecutionEnvironment,
    *,
    fingerprint: str,
    binaries: tuple[str, ...],
    role_homes: tuple[str, ...],
    agent_prompt: str,
) -> str:
    """Check an existing container without creating paths or copying seed files."""
    return (
        _preamble(
            environment,
            fingerprint=fingerprint,
            binaries=binaries,
            role_homes=role_homes,
            agent_prompt=agent_prompt,
        )
        + r"""
if [ ! -f "$READY_MARKER" ] || [ "$(cat -- "$READY_MARKER")" != "$FINGERPRINT" ]; then
  fail "Docker runner readiness fingerprint does not match"
fi
"""
        + _checks()
        + r"""
test -d "$WORKSPACE/target" || fail "workspace target is missing"
"""
    )
