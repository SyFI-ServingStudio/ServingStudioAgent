"""Catalog-file capabilities layered over an explicit set of selectable models."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from .base import Model

LOG = logging.getLogger(__name__)


class FileCatalog:
    def __init__(
        self,
        models: tuple[Model, ...],
        paths: tuple[Path, ...],
    ):
        self.models = models
        self.paths = paths

    def __call__(self) -> tuple[Model, ...]:
        entries = {}
        for path in self.paths:
            try:
                payload = json.loads(path.read_text())
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                LOG.warning("Could not read model catalog: %s", path)
                continue
            raw_models = payload.get("models") if isinstance(payload, dict) else None
            if isinstance(raw_models, list):
                entries = {
                    entry["slug"]: entry
                    for entry in raw_models
                    if isinstance(entry, dict) and isinstance(entry.get("slug"), str)
                }
                break
        result = []
        for model in self.models:
            entry = entries.get(model.model_id, {})
            raw_tiers = entry.get("additional_speed_tiers", [])
            tiers = (
                tuple(tier for tier in raw_tiers if isinstance(tier, str) and tier)
                if isinstance(raw_tiers, list)
                else ()
            )
            label = entry.get("display_name")
            result.append(
                replace(
                    model,
                    service_tiers=tuple(dict.fromkeys((*model.service_tiers, *tiers))),
                    label=label if isinstance(label, str) and label else model.label,
                )
            )
        return tuple(result)
