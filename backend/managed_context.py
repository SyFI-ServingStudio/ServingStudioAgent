"""Short-lived capabilities for Launcher calls made inside managed Agent turns."""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass

from .codex_runtime.config import (
    CODEX_DOCKER_CODEX_ROOT,
    codex_home_for,
)

MANAGED_CONTEXT_FILENAME = "managed-run.json"
MANAGED_CONTEXT_CONTAINER_PATH = f"{CODEX_DOCKER_CODEX_ROOT}/{MANAGED_CONTEXT_FILENAME}"
MANAGED_BACKEND_URL = os.environ.get(
    "VIBESIM_MANAGED_BACKEND_URL",
    "http://host.docker.internal:8765",
).rstrip("/")


@dataclass(frozen=True, slots=True)
class Capability:
    token: str
    workspace_id: str
    conversation_id: str
    turn_id: str
    role: str
    expires_at: float


class CapabilityRegistry:
    """In-memory authority for one backend process.

    A backend restart intentionally invalidates outstanding Launcher
    capabilities. Managed Launcher calls then fail before writing run artifacts,
    which is safer than silently losing provenance.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._capabilities: dict[str, Capability] = {}

    def issue(
        self,
        *,
        workspace_id: str,
        conversation_id: str,
        turn_id: str,
        role: str,
        ttl_seconds: float = 7200,
    ) -> Capability:
        now = time.time()
        capability = Capability(
            token=secrets.token_urlsafe(32),
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            role=role,
            expires_at=now + ttl_seconds,
        )
        with self._lock:
            self._purge_expired(now)
            self._capabilities[capability.token] = capability
        return capability

    def authorize(self, token: str) -> Capability | None:
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            capability = self._capabilities.get(token)
            if capability is None or capability.expires_at <= now:
                return None
            return capability

    def revoke_turn(self, workspace_id: str, turn_id: str) -> None:
        with self._lock:
            expired = [
                token
                for token, capability in self._capabilities.items()
                if capability.workspace_id == workspace_id
                and capability.turn_id == turn_id
            ]
            for token in expired:
                self._capabilities.pop(token, None)

    def _purge_expired(self, now: float) -> None:
        expired = [
            token
            for token, capability in self._capabilities.items()
            if capability.expires_at <= now
        ]
        for token in expired:
            self._capabilities.pop(token, None)


capabilities = CapabilityRegistry()


def write_managed_context(
    *,
    workspace_id: str,
    conversation_id: str,
    turn_id: str,
    role: str,
) -> Capability:
    capability = capabilities.issue(
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        role=role,
    )
    context_path = (
        codex_home_for(workspace_id, conversation_id) / MANAGED_CONTEXT_FILENAME
    )
    context_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "backend_url": MANAGED_BACKEND_URL,
        "capability_token": capability.token,
        "expires_at": capability.expires_at,
    }
    temporary = context_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
    temporary.chmod(0o600)
    temporary.replace(context_path)
    return capability


def remove_managed_context(workspace_id: str, conversation_id: str) -> None:
    context_path = (
        codex_home_for(workspace_id, conversation_id) / MANAGED_CONTEXT_FILENAME
    )
    context_path.unlink(missing_ok=True)


def bearer_token(authorization: str | None) -> str:
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        return ""
    return authorization[len(prefix) :].strip()
