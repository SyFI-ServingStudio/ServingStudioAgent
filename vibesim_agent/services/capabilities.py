"""Process-local managed-call authority and atomic per-conversation context files."""

from __future__ import annotations

import contextlib
import json
import math
import os
import secrets
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..providers.base import AgentRequest

MANAGED_CONTEXT_FILENAME = "managed-run.json"


@dataclass(frozen=True, slots=True)
class Capability:
    token: str = field(repr=False)
    workspace_id: str
    conversation_id: str
    turn_id: str
    role: str
    expires_at: float


class CapabilityRegistry:
    """One process's authority; clock returns epoch seconds for wire expiry.

    Restarting the backend creates an empty registry and invalidates old tokens.
    A persisted context file is not itself an authorization source.
    """

    def __init__(self, *, clock: Callable[[], float]):
        self.clock = clock
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
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("capability lifetime must be finite and positive")
        with self._lock:
            now = self.clock()
            expires_at = now + ttl_seconds
            if not math.isfinite(now) or not math.isfinite(expires_at):
                raise ValueError("capability clock and expiry must be finite")
            self._purge_expired(now)
            token = secrets.token_urlsafe(32)
            while token in self._capabilities:
                token = secrets.token_urlsafe(32)
            capability = Capability(
                token, workspace_id, conversation_id, turn_id, role, expires_at
            )
            self._capabilities[token] = capability
            return capability

    def authorize(self, token: str) -> Capability | None:
        with self._lock:
            now = self.clock()
            if not math.isfinite(now):
                raise ValueError("capability clock must be finite")
            self._purge_expired(now)
            return self._capabilities.get(token)

    def revoke(self, token: str) -> None:
        with self._lock:
            self._capabilities.pop(token, None)

    def revoke_turn(self, workspace_id: str, turn_id: str) -> None:
        with self._lock:
            tokens = [
                token
                for token, capability in self._capabilities.items()
                if capability.workspace_id == workspace_id
                and capability.turn_id == turn_id
            ]
            for token in tokens:
                self._capabilities.pop(token, None)

    def _purge_expired(self, now: float) -> None:
        tokens = [
            token
            for token, capability in self._capabilities.items()
            if capability.expires_at <= now
        ]
        for token in tokens:
            self._capabilities.pop(token, None)


class ManagedContext:
    def __init__(
        self,
        registry: CapabilityRegistry,
        path: Callable[[str, str], Path],
    ):
        self.registry = registry
        self.path = path
        self._lock = threading.RLock()

    def write(self, request: AgentRequest) -> Capability:
        with self._lock:
            capability = self.registry.issue(
                workspace_id=request.workspace_id,
                conversation_id=request.conversation_id,
                turn_id=request.turn_id,
                role=request.role.value,
            )
            temporary: Path | None = None
            descriptor: int | None = None
            try:
                path = self.path(request.workspace_id, request.conversation_id)
                if not path.is_absolute():
                    raise ValueError("managed context path must be absolute")
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                payload = {
                    "schema_version": 1,
                    "managed_jobs_api": "agent-v1",
                    # Read off the turn, not fixed at construction: the default
                    # `host.docker.internal` resolves only inside a container,
                    # and the same backend needs another name on the host.
                    "backend_url": request.execution.managed_backend_url.rstrip("/"),
                    "capability_token": capability.token,
                    "expires_at": capability.expires_at,
                }
                descriptor, name = tempfile.mkstemp(
                    dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
                )
                temporary = Path(name)
                stream = os.fdopen(descriptor, "w", encoding="utf-8")
                descriptor = None
                with stream:
                    os.fchmod(stream.fileno(), 0o600)
                    json.dump(payload, stream, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
                return capability
            except BaseException:
                self.registry.revoke(capability.token)
                raise
            finally:
                if descriptor is not None:
                    with contextlib.suppress(OSError):
                        os.close(descriptor)
                if temporary is not None:
                    with contextlib.suppress(OSError):
                        temporary.unlink(missing_ok=True)

    def remove(self, workspace_id: str, conversation_id: str) -> None:
        """Remove the published file; the turn owner separately revokes its tokens."""
        with self._lock:
            path = self.path(workspace_id, conversation_id)
            if not path.is_absolute():
                raise ValueError("managed context path must be absolute")
            path.unlink(missing_ok=True)


def bearer_token(authorization: str | None) -> str:
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        return ""
    return authorization[len(prefix) :].strip()
