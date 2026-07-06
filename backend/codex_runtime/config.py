"""Runtime configuration and stable path helpers for Docker-backed Codex calls."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
UI_DIR = Path(__file__).resolve().parents[2]
MAIN_DIR = WORKSPACE / "main"
WORKSPACES_DIR = UI_DIR / "workspaces"
PROMPTS_DIR = UI_DIR / "backend" / "prompts"

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
AGENTS_PROMPT_DEFAULT = "AGENTS.md"
AGENTS_PROMPT_AUTONOMOUS = "AGENTS.autonomous.md"

EXECUTION_MODES = ("read-only", "workspace-write", "danger-full-access")
SANDBOX_MODES = EXECUTION_MODES
DEFAULT_SANDBOX = "workspace-write"

LOG = logging.getLogger("mlsim_ui.codex_runtime")

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MAIN_LOCK_SHA = os.environ.get("CODEX_MAIN_LOCK_SHA") or _sha256_file(MAIN_DIR / "uv.lock")


def agents_prompt_name(autonomous: bool) -> str:
    return AGENTS_PROMPT_AUTONOMOUS if autonomous else AGENTS_PROMPT_DEFAULT


def prompt_fingerprint(*, autonomous: bool = False) -> str:
    """Hash role prompts/schema so stale Codex sessions are not resumed."""
    digest = hashlib.sha256()
    for name in (
        agents_prompt_name(autonomous),
        "orchestrator.txt",
        "implementer.txt",
        "orchestrator.schema.json",
    ):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((PROMPTS_DIR / name).read_bytes())
        digest.update(b"\0")
    digest.update(CODEX_MODEL.encode("utf-8"))
    digest.update(b"\0autonomous=")
    digest.update(str(autonomous).encode("utf-8"))
    return digest.hexdigest()[:16]

def workspace_main_for(conversation_id: str) -> Path:
    return WORKSPACES_DIR / conversation_id / "main"

def codex_home_for(conversation_id: str) -> Path:
    return WORKSPACES_DIR / conversation_id / "codex-home"

def container_name(conversation_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", conversation_id)[:48]
    return f"mlsim-ui-{safe}"
