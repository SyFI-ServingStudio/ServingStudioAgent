"""Configuration is explicit, validates inputs, and does not disclose secrets."""

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.settings import (
    ConfigurationError,
    ProviderEnvironment,
    ProviderSettings,
    environment_reference,
    load_settings,
)


class SettingsTests(unittest.TestCase):
    def test_registered_provider_uses_derived_environment_without_core_branch(self):
        profile = ProviderEnvironment(
            "example",
            ProviderSettings(model="base", effort="low"),
            secret_names=("EXAMPLE_API_KEY",),
        )
        settings = load_settings(
            environment={
                "VIBESIM_PROVIDER_EXAMPLE_MODEL": "other",
                "EXAMPLE_API_KEY": "private-value",
            },
            providers=[profile],
        )
        self.assertEqual(settings.providers["example"].model, "other")
        self.assertEqual(
            settings.secrets["EXAMPLE_API_KEY"].get_secret_value(), "private-value"
        )
        self.assertNotIn("private-value", repr(settings))
        self.assertNotIn("private-value", settings.model_dump_json())
        reference = environment_reference([profile])
        self.assertIn("VIBESIM_PROVIDER_EXAMPLE_MODEL", reference)
        self.assertNotIn("VIBESIM_PROVIDER_EXAMPLE_HOME", reference)

    def test_invalid_inputs_identify_keys_without_values(self):
        for key, value in (
            ("VIBESIM_AGENT_IDLE_TIMEOUT", "nan"),
            ("VIBESIM_AGENT_PORT", "70000"),
            ("VIBESIM_RUNNER_UID", "not-an-id"),
            ("VIBESIM_AGENT_ANALYZER_BASE_URL", "http://secret:credential@host"),
            ("VIBESIM_AGENT_ANALYZER_BASE_URL", "http://host/api/analyzer/v1"),
            ("VIBESIM_AGENT_MAIN_DIR", "relative"),
        ):
            with self.subTest(key=key), self.assertRaises(ConfigurationError) as caught:
                load_settings(environment={key: value})
            self.assertIn(key, str(caught.exception))
            self.assertNotIn(value, str(caught.exception))

    def test_settings_are_independent_and_loading_creates_no_state(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = load_settings(
                environment={"VIBESIM_AGENT_WORKSPACES_ROOT": str(root / "first")},
                repo_root=root,
            )
            second = load_settings(
                environment={"VIBESIM_AGENT_WORKSPACES_ROOT": str(root / "second")},
                repo_root=root,
            )
            self.assertNotEqual(
                first.agent.workspaces_root, second.agent.workspaces_root
            )
            self.assertEqual(list(root.iterdir()), [])

    def test_import_neither_reads_environment_nor_creates_runtime_files(self):
        script = """
import os
import sys
from pathlib import Path
from unittest.mock import patch
# Pydantic lazily imports dependencies, including sysconfig, at first model build.
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator
class WarmDependencies(BaseModel):
    value: str
WarmDependencies(value="ready")
original_getitem = type(os.environ).__getitem__
def guarded_getitem(environment, key):
    caller = sys._getframe(1)
    while caller and caller.f_globals.get("__name__") in {"os", "collections.abc", "_collections_abc"}:
        caller = caller.f_back
    # Pydantic checks this switch for every model; only its own read is allowed.
    if (key == "PYDANTIC_DISABLE_PLUGINS" and caller is not None
            and caller.f_globals.get("__name__") == "pydantic.plugin._loader"):
        return original_getitem(environment, key)
    raise AssertionError("application environment read: " + key)
with patch.dict(os.environ, {}, clear=True):
    with patch.object(type(os.environ), "__getitem__", new=guarded_getitem), \\
         patch.object(Path, "mkdir", side_effect=AssertionError("filesystem write")), \\
         patch.object(Path, "write_text", side_effect=AssertionError("filesystem write")):
        import vibesim_agent.settings
        import vibesim_agent.domain.roles
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_analyzer_aliases_and_origin_normalization(self):
        for source, expected in (
            ("host", "external"), ("local", "workspace"),
            (" HOST ", "external"), ("Workspace", "workspace"),
        ):
            settings = load_settings(environment={
                "VIBESIM_AGENT_ANALYZER_SOURCE": source,
                "VIBESIM_AGENT_ANALYZER_BASE_URL": "https://analyzer:8787/",
                "VIBESIM_AGENT_NAMING_BASE_URL": "https://naming/api/v1/",
            })
            self.assertEqual(settings.agent.analyzer_source, expected)
            self.assertEqual(settings.agent.analyzer_base_url, "https://analyzer:8787")
            self.assertEqual(settings.agent.naming_base_url, "https://naming/api/v1")

    def test_host_paths_expand_using_explicit_environment(self):
        profile = ProviderEnvironment(
            "example", ProviderSettings(model="base", effort="low"), accepts_home=True
        )
        settings = load_settings(environment={
            "HOME": "/host-home",
            "VIBESIM_AGENT_MAIN_DIR": "~/source",
            "VIBESIM_AGENT_WORKSPACES_ROOT": "~/workspaces",
            "HF_HOME": "~/.cache/huggingface",
            "VIBESIM_PROVIDER_EXAMPLE_HOME": "~/.example",
            "VIBESIM_RUNNER_USER": " alice ",
        }, providers=[profile])
        self.assertEqual(settings.agent.main_dir, Path("/host-home/source"))
        self.assertEqual(settings.agent.workspaces_root, Path("/host-home/workspaces"))
        self.assertEqual(settings.container.hf_home, Path("/host-home/.cache/huggingface"))
        self.assertEqual(settings.providers["example"].home, Path("/host-home/.example"))
        self.assertEqual(settings.container.user, "alice")
        self.assertEqual(settings.container.home, Path("/home/alice"))
        self.assertEqual(settings.container.image, "vibesim-agent-runner:alice")
        for key in ("HOME", "UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR"):
            with self.subTest(key=key), self.assertRaises(ConfigurationError):
                load_settings(environment={f"VIBESIM_RUNNER_{key}": "~/container"})

    def test_no_home_option_for_environment_authenticated_provider(self):
        profile = ProviderEnvironment(
            "example", ProviderSettings(model="base", effort="low")
        )
        with self.assertRaisesRegex(ConfigurationError, "HOME is not supported"):
            load_settings(
                environment={"VIBESIM_PROVIDER_EXAMPLE_HOME": "/private"},
                providers=[profile],
            )

    def test_duplicate_or_ambiguous_provider_ids_are_rejected(self):
        profile = ProviderEnvironment(
            "example", ProviderSettings(model="base", effort="low")
        )
        with self.assertRaisesRegex(ConfigurationError, "Duplicate provider"):
            load_settings(environment={}, providers=[profile, profile])
        bad = ProviderEnvironment("EXAMPLE", profile.defaults)
        with self.assertRaises(ConfigurationError):
            load_settings(environment={}, providers=[bad])
