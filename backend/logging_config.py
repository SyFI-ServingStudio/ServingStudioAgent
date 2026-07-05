"""Structured logging for the MLSim UI backend."""

from __future__ import annotations

import json
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "backend.log"


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event_fields = getattr(record, "event_fields", None)
        if isinstance(event_fields, dict):
            payload.update(event_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging() -> None:
    """Install stdout and rotating-file JSON logs once per process."""
    root = logging.getLogger("mlsim_ui")
    if getattr(root, "_mlsim_configured", False):
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = JsonLineFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root.setLevel(logging.INFO)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    root.propagate = False
    root._mlsim_configured = True  # type: ignore[attr-defined]


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"event_fields": {"event": event, **fields}})


def compact_text(text: str, limit: int = 240) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."
