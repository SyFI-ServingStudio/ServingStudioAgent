import json
from pathlib import Path


def write_minimal_providers(repo_root: Path, *, home: str = "~/.codex") -> Path:
    path = repo_root / "providers.yaml"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "gpt": {
                        "adapter": "codex",
                        "home": home,
                        "default_model": "test-model",
                        "default_effort": "high",
                        "models": {
                            "test-model": {"efforts": ["low", "high"]}
                        },
                    }
                },
                "defaults": {
                    "orchestrator": "gpt",
                    "implementer": "gpt",
                    "assistant": "gpt",
                },
            }
        )
    )
    return path
