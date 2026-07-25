"""In-memory + JSON-file conversation store.

A conversation holds displayed message history, selected execution mode, and the
Codex session ids for each role. Persistence is a single ``conversations.json``
next to the backend package, written atomically so a restart reloads prior chats.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

STORE_PATH = Path(__file__).resolve().parents[1] / "conversations.json"


class Store:
    def __init__(self, path: Path = STORE_PATH) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._conversations: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text("utf-8"))
            for conv in data.get("conversations", []):
                if "id" in conv:
                    conv.setdefault("messages", [])
                    conv.setdefault("codex_sessions", {})
                    conv.setdefault("autonomous", False)
                    conv.setdefault("peer_workspace", None)
                    self._conversations[conv["id"]] = conv
        except (json.JSONDecodeError, OSError):
            # Corrupt/unreadable store: start empty rather than crash the server.
            self._conversations = {}

    def _save(self) -> None:
        payload = {"conversations": list(self._conversations.values())}
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self._path)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            items = sorted(
                self._conversations.values(),
                key=lambda c: c.get("updated_at", 0),
                reverse=True,
            )
            return [
                {"id": c["id"], "title": c.get("title", "New chat"), "updated_at": c.get("updated_at", 0)}
                for c in items
            ]

    def get(self, cid: str) -> dict[str, Any] | None:
        with self._lock:
            conv = self._conversations.get(cid)
            return json.loads(json.dumps(conv)) if conv else None

    def create(
        self,
        cid: str,
        sandbox: str,
        prompt_fingerprint: str | None = None,
        *,
        autonomous: bool = False,
        peer_workspace: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        conv = {
            "id": cid,
            "title": "New chat",
            "sandbox": sandbox,
            "autonomous": autonomous,
            "prompt_fingerprint": prompt_fingerprint,
            # Co-evolution: host path to the caller's (vibe-serve) candidate
            # workspace to bind read-only at /candidate when this conversation's
            # container launches. None -> no reverse mount.
            "peer_workspace": peer_workspace,
            "codex_sessions": {},
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._conversations[cid] = conv
            self._save()
            return dict(conv)

    def update_runtime_settings(self, cid: str, *, sandbox: str, autonomous: bool) -> None:
        with self._lock:
            conv = self._conversations.get(cid)
            if conv is None:
                return
            conv["sandbox"] = sandbox
            conv["autonomous"] = autonomous
            conv["updated_at"] = time.time()
            self._save()

    def add_message(self, cid: str, role: str, content: str, **metadata: Any) -> None:
        with self._lock:
            conv = self._conversations.get(cid)
            if conv is None:
                return
            message = {"role": role, "content": content, "ts": time.time()}
            for key, value in metadata.items():
                if value is not None:
                    message[key] = value
            conv["messages"].append(message)
            if role == "user" and conv.get("title", "New chat") == "New chat":
                first_line = content.strip().splitlines()[0] if content.strip() else ""
                conv["title"] = (first_line[:48] or "New chat")
            conv["updated_at"] = time.time()
            self._save()

    def sessions_for_prompt(self, cid: str, prompt_fingerprint: str) -> dict[str, str]:
        """Return role sessions, dropping them when role prompts/schema changed."""
        with self._lock:
            conv = self._conversations.get(cid)
            if conv is None:
                return {}
            if conv.get("prompt_fingerprint") != prompt_fingerprint:
                conv["prompt_fingerprint"] = prompt_fingerprint
                conv["codex_sessions"] = {}
                self._save()
            return dict(conv.get("codex_sessions") or {})

    def set_codex_session(self, cid: str, role: str, session_id: str | None) -> None:
        if not session_id:
            return
        with self._lock:
            conv = self._conversations.get(cid)
            if conv is None:
                return
            sessions = conv.setdefault("codex_sessions", {})
            if sessions.get(role) != session_id:
                sessions[role] = session_id
                self._save()

    def delete(self, cid: str) -> None:
        with self._lock:
            self._conversations.pop(cid, None)
            self._save()
