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
ANALYZER_MCP_DIR = UI_DIR / "backend" / "analyzer_evidence_mcp"
ANALYZER_MCP_CONTAINER_DIR = "/opt/vibesim/analyzer-evidence-mcp"
ANALYZER_MCP_PYTHON = "/opt/vibesim-analyzer-mcp-venv/bin/python"
ANALYZER_MCP_SOURCE = os.environ.get("ANALYZER_MCP_SOURCE", "external").strip()
ANALYZER_MCP_BASE_URL = os.environ.get(
    "ANALYZER_MCP_BASE_URL",
    "http://host.docker.internal:8787",
).rstrip("/")

CODEX_DOCKER_IMAGE = os.environ.get(
    "CODEX_DOCKER_IMAGE", "vibesim-ui-codex-runner:latest"
)
CODEX_MODEL = os.environ.get("CODEX_MODEL", "gpt-5.6-sol")
# Codex reasoning effort, passed per call as `-c model_reasoning_effort=...`.
CODEX_REASONING_EFFORT = os.environ.get("CODEX_REASONING_EFFORT", "xhigh")
# Bearer token gating the agent-facing HTTP API (/api/agent/*, /api/eval).
# Unset -> no auth, so local dev and the same-host eval harness keep working.
# Set it when exposing the backend to cross-machine agents.
VIBESIM_API_TOKEN = os.environ.get("VIBESIM_API_TOKEN", "").strip()
CODEX_IDLE_TIMEOUT = float(
    os.environ.get("CODEX_IDLE_TIMEOUT", os.environ.get("CODEX_TURN_TIMEOUT", "600"))
)
CODEX_DOCKER_GPUS = os.environ.get("CODEX_DOCKER_GPUS", "all").strip()
CODEX_DOCKER_UID = int(os.environ.get("CODEX_DOCKER_UID", str(os.getuid())))
CODEX_DOCKER_GID = int(os.environ.get("CODEX_DOCKER_GID", str(os.getgid())))
CODEX_DOCKER_USER = (
    os.environ.get("CODEX_DOCKER_USER") or os.environ.get("USER") or "codex"
)
CODEX_DOCKER_HOME = os.environ.get("CODEX_DOCKER_HOME", f"/home/{CODEX_DOCKER_USER}")
CODEX_DOCKER_AUTH_DIR = f"{CODEX_DOCKER_HOME}/.codex"
CODEX_DOCKER_DG_USE_LOCAL_VERSION = os.environ.get(
    "CODEX_DOCKER_DG_USE_LOCAL_VERSION", "0"
)
CODEX_DOCKER_UV_PROJECT_ENVIRONMENT = os.environ.get(
    "CODEX_DOCKER_UV_PROJECT_ENVIRONMENT",
    "/opt/vibesim-venv",
)
CODEX_DOCKER_UV_CACHE_DIR = os.environ.get(
    "CODEX_DOCKER_UV_CACHE_DIR", "/opt/vibesim-uv-cache"
)
_host_hf_home = os.environ.get("HF_HOME", "").strip()
HOST_HF_HOME = Path(_host_hf_home).expanduser() if _host_hf_home else None
CODEX_DOCKER_HF_HOME = "/model"
CONTAINER_RUNTIME_VERSION = os.environ.get(
    "CODEX_RUNNER_IMAGE_VERSION", "prebuilt-codex-runner-v9"
)
ORCHESTRATOR_SCHEMA_IN_CONTAINER = "/workspace/.codex/orchestrator.schema.json"
AGENTS_PROMPT_DEFAULT = "AGENTS.md"
AGENTS_PROMPT_AUTONOMOUS = "AGENTS.autonomous.md"

EXECUTION_MODES = ("read-only", "workspace-write", "danger-full-access")
SANDBOX_MODES = EXECUTION_MODES
DEFAULT_SANDBOX = "workspace-write"

LOG = logging.getLogger("vibesim_ui.codex_runtime")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MAIN_LOCK_SHA = os.environ.get("CODEX_MAIN_LOCK_SHA") or _sha256_file(
    MAIN_DIR / "uv.lock"
)
MAIN_BUILD_SHA = os.environ.get("CODEX_MAIN_BUILD_SHA", "").strip()


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
    return f"vibesim-ui-{safe}"
