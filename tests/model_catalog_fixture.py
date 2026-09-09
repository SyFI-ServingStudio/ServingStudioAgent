"""Explicit model capabilities for tests that select fast service or max effort."""

from unittest import TestCase
from unittest.mock import patch

from backend.codex_runtime import config


def install_model_catalog(test_case: TestCase) -> None:
    catalog = {
        model: {
            "slug": model,
            "supported_reasoning_levels": [
                {"effort": effort} for effort in ("low", "medium", "high", "xhigh", "max")
            ],
            "additional_speed_tiers": ["fast"],
        }
        for model in ("gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra")
    }
    test_case.enterContext(patch.object(config, "_MODEL_REGISTRY_CACHE", {}))
    test_case.enterContext(
        patch.object(
            config,
            "_catalog_models",
            side_effect=lambda family: catalog if family.family_id == "gpt" else {},
        )
    )
