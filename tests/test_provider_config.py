"""Named-provider configuration uses references, strict YAML, and isolated inputs."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vibesim_agent.bootstrap import configuration
from vibesim_agent.provider_config import load_provider_config
from vibesim_agent.settings import (
    ConfigurationError,
    ProviderEnvironment,
    ProviderSettings,
    load_settings,
)


class ProviderConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "providers.yaml"
        self.document = {
            "version": 1,
            "providers": {
                "work": {
                    "adapter": "claude",
                    "label": "Work connection",
                    "model": "claude-sonnet-5",
                    "effort": "high",
                    "base_url": "https://work.example.test/v1/",
                    "environment": {"ANTHROPIC_API_KEY": "WORK_KEY"},
                },
                "personal": {
                    "adapter": "claude",
                    "model": "claude-sonnet-5",
                    "effort": "medium",
                    "environment": {"ANTHROPIC_AUTH_TOKEN": "PERSONAL_TOKEN"},
                },
                "code": {
                    "adapter": "codex",
                    "model": "gpt-5.6-sol",
                    "effort": "high",
                    "home": "~/.codex-work",
                    "environment": {"CUSTOM_API_KEY": "CODE_KEY"},
                },
            },
            "defaults": {
                "orchestrator": "work",
                "implementer": "code",
                "assistant": "personal",
            },
        }

    def write(self, document=None):
        self.path.write_text(
            json.dumps(self.document if document is None else document)
        )
        return self.path

    def load(self, document=None, **environment):
        return load_provider_config(
            self.write(document), environment={"HOME": str(self.root), **environment}
        )

    def test_repository_file_is_discovered_without_consulting_cwd(self):
        self.write()
        with patch.object(Path, "cwd", side_effect=AssertionError("cwd lookup")):
            configured = load_settings(
                repo_root=self.root, environment={"HOME": str(self.root)}
            )
        self.assertEqual(set(configured.providers), set(self.document["providers"]))
        self.assertEqual(configured.role_providers, self.document["defaults"])

    def test_explicit_file_overrides_even_invalid_repository_file(self):
        explicit = self.root / "explicit.yaml"
        explicit.write_text(json.dumps(self.document))
        self.path.mkdir()
        configured = load_settings(
            repo_root=self.root,
            environment={
                "VIBESIM_AGENT_PROVIDERS_FILE": str(explicit),
                "HOME": str(self.root),
            },
        )
        self.assertEqual(set(configured.providers), set(self.document["providers"]))

    def test_startup_configuration_discovers_repository_role_defaults(self):
        self.write()
        configured = configuration(
            repo_root=self.root, environment={"HOME": str(self.root)}
        )
        self.assertEqual(configured.agent.repo_root, self.root)
        self.assertEqual(configured.role_providers, self.document["defaults"])
        self.assertEqual(set(configured.connections), set(self.document["providers"]))

    def test_invalid_repository_file_never_falls_back(self):
        for kind in ("malformed", "directory", "dangling_symlink"):
            with self.subTest(kind=kind):
                if kind == "malformed":
                    self.path.write_text("[invalid")
                elif kind == "directory":
                    self.path.mkdir()
                else:
                    self.path.symlink_to(self.root / "missing.yaml")
                try:
                    with self.assertRaises(ConfigurationError):
                        load_settings(repo_root=self.root, environment={})
                finally:
                    if kind == "directory":
                        self.path.rmdir()
                    else:
                        self.path.unlink()

    def test_discovered_file_rejects_conflicting_provider_environment(self):
        self.write()
        with self.assertRaisesRegex(ConfigurationError, "conflicts"):
            load_settings(
                repo_root=self.root,
                environment={"VIBESIM_PROVIDER_WORK_MODEL": "override"},
            )

    def test_named_connections_resolve_only_their_references_without_source_writes(
        self,
    ):
        self.write()
        before = self.path.read_bytes(), self.path.stat().st_mtime_ns
        result = load_provider_config(
            self.path,
            environment={
                "HOME": str(self.root),
                "WORK_KEY": "work-secret",
                "PERSONAL_TOKEN": "personal-secret",
                "CODE_KEY": "code-secret",
                "ANTHROPIC_API_KEY": "unrelated-secret",
            },
        )
        self.assertEqual(set(result.providers), {"work", "personal", "code"})
        self.assertEqual(
            result.connections["work"].environment, {"ANTHROPIC_API_KEY": "WORK_KEY"}
        )
        self.assertEqual(
            result.connections["work"].base_url, "https://work.example.test/v1"
        )
        self.assertIsNone(result.connections["personal"].base_url)
        self.assertIsNone(result.connections["work"].session_identity)
        self.assertEqual(result.providers["code"].home, self.root / ".codex-work")
        self.assertEqual(result.role_providers, self.document["defaults"])
        self.assertEqual(
            set(result.secrets), {"WORK_KEY", "PERSONAL_TOKEN", "CODE_KEY"}
        )
        self.assertEqual(result.secrets["WORK_KEY"].get_secret_value(), "work-secret")
        self.assertNotIn("work-secret", repr(result))
        self.assertEqual(before, (self.path.read_bytes(), self.path.stat().st_mtime_ns))

    def test_missing_reference_does_not_inherit_global_auth_or_ambient_environment(
        self,
    ):
        with patch.dict(os.environ, {"WORK_KEY": "ambient-secret"}):
            result = self.load(ANTHROPIC_API_KEY="global-secret")
        self.assertEqual(result.secrets, {})
        self.assertIsNone(result.connections["personal"].label)

    def test_models_default_to_none_and_explicit_lists_preserve_exact_ids(self):
        result = self.load()
        self.assertIsNone(result.connections["work"].models)
        document = json.loads(json.dumps(self.document))
        document["providers"]["work"]["models"] = ["claude-sonnet-5", "claude-opus-5"]
        result = self.load(document)
        self.assertEqual(
            result.connections["work"].models, ("claude-sonnet-5", "claude-opus-5")
        )
        self.assertEqual(result.providers["work"].model, "claude-sonnet-5")

    def test_models_reject_empty_duplicates_missing_default_and_wrong_types(self):
        values = [
            [],
            None,
            "claude-sonnet-5",
            {},
            ["claude-sonnet-5", "claude-sonnet-5"],
            ["claude-opus-5"],
            ["claude-sonnet-5", ""],
            ["claude-sonnet-5", "  "],
            ["claude-sonnet-5", 7],
            ["claude-sonnet-5", {}],
            ["sonnet"],
        ]
        for index, models in enumerate(values):
            with self.subTest(index=index):
                document = json.loads(json.dumps(self.document))
                document["providers"]["work"]["models"] = models
                with self.assertRaisesRegex(
                    ConfigurationError, "Invalid providers file configuration"
                ):
                    self.load(document)

    def test_claude_home_is_an_alternative_to_one_auth_reference(self):
        document = json.loads(json.dumps(self.document))
        work = document["providers"]["work"]
        work.pop("environment")
        work.update(home="~/.claudeme", session_identity="work-account")
        result = self.load(document)
        self.assertEqual(result.providers["work"].home, self.root / ".claudeme")
        self.assertEqual(result.connections["work"].session_identity, "work-account")
        self.assertEqual(result.connections["work"].environment, {})
        self.assertFalse((self.root / ".claudeme").exists())
        work["environment"] = {"ANTHROPIC_API_KEY": "WORK_KEY"}
        with self.assertRaises(ConfigurationError):
            self.load(document)
        work.pop("environment")
        work.pop("home")
        with self.assertRaises(ConfigurationError):
            self.load(document)

    def test_yaml_replaces_legacy_profiles_but_keeps_naming_secret(self):
        old = ProviderEnvironment(
            "legacy", ProviderSettings(model="old", effort="high")
        )
        configured = load_settings(
            repo_root=self.root,
            providers=(old,),
            environment={
                "VIBESIM_AGENT_PROVIDERS_FILE": str(self.write()),
                "HOME": str(self.root),
                "WORK_KEY": "private",
                "OPENROUTER_API_KEY": "naming-private",
            },
        )
        self.assertEqual(configured.providers["work"].model, "claude-sonnet-5")
        self.assertNotIn("legacy", configured.providers)
        self.assertEqual(set(configured.secrets), {"WORK_KEY", "OPENROUTER_API_KEY"})
        self.assertEqual(configured.role_providers, self.document["defaults"])

    def test_yaml_rejects_simultaneous_legacy_provider_overrides(self):
        for value in ("SECRET_DO_NOT_PRINT", ""):
            with (
                self.subTest(empty=not value),
                self.assertRaisesRegex(
                    ConfigurationError, "conflicts with VIBESIM_PROVIDER"
                ) as caught,
            ):
                load_settings(
                    repo_root=self.root,
                    environment={
                        "VIBESIM_AGENT_PROVIDERS_FILE": str(self.write()),
                        "VIBESIM_PROVIDER_WORK_MODEL": value,
                        "HOME": str(self.root),
                    },
                )
            self.assertNotIn("SECRET_DO_NOT_PRINT", str(caught.exception))

    def test_absent_yaml_keeps_legacy_loading(self):
        profile = ProviderEnvironment(
            "legacy",
            ProviderSettings(model="old", effort="high"),
            secret_names=("LEGACY_KEY",),
        )
        result = load_settings(
            repo_root=self.root,
            providers=(profile,),
            environment={
                "VIBESIM_PROVIDER_LEGACY_MODEL": "changed",
                "LEGACY_KEY": "old-secret",
            },
        )
        self.assertEqual(result.providers["legacy"].model, "changed")
        self.assertEqual(result.connections, {})
        self.assertEqual(result.role_providers, {})
        self.assertEqual(result.secrets["LEGACY_KEY"].get_secret_value(), "old-secret")

    def test_duplicate_keys_unsafe_tags_aliases_and_invalid_yaml_are_redacted(self):
        bodies = [
            "version: 1\nversion: 1\n",
            "version: 1\nproviders:\n  work: {adapter: claude, adapter: codex}\n",
            "secret: !!python/object/apply:os.system ['DO_NOT_RUN']",
            "providers: &private {work: value}\ndefaults: *private\n",
            "SECRET_DO_NOT_PRINT: [unterminated",
            "- SECRET_DO_NOT_PRINT",
            "null",
            "version: true",
            "---\nversion: 1\n---\nversion: 1",
        ]
        for body in bodies:
            with self.subTest(body_index=bodies.index(body)):
                self.path.write_text(body)
                with self.assertRaises(ConfigurationError) as caught:
                    load_provider_config(self.path, environment={})
                self.assertEqual(
                    str(caught.exception), "Invalid providers file configuration"
                )

    def test_invalid_shapes_defaults_endpoints_and_ids_are_rejected(self):
        mutations = [
            lambda d: d.update(version=True),
            lambda d: d.update(version=2),
            lambda d: d.update(unknown="SECRET_DO_NOT_PRINT"),
            lambda d: d.update(providers={}),
            lambda d: d["defaults"].pop("assistant"),
            lambda d: d["defaults"].update(assistant="missing"),
            lambda d: d["defaults"].update(assistant=[]),
            lambda d: d["providers"].update({"BAD-ID": d["providers"].pop("work")}),
            lambda d: d["providers"]["work"].update(api_key="SECRET_DO_NOT_PRINT"),
            lambda d: d["providers"]["work"].update(model=7),
            lambda d: d["providers"]["work"].update(
                base_url="https://user:SECRET_DO_NOT_PRINT@example.test"
            ),
            lambda d: d["providers"]["work"].update(
                base_url="https://example.test/?key=SECRET_DO_NOT_PRINT"
            ),
            lambda d: d["providers"]["work"].update(
                base_url="https://example.test:99999"
            ),
            lambda d: d["providers"]["code"].update(base_url="https://example.test"),
            lambda d: d["providers"]["code"].update(home="relative-home"),
            lambda d: d["providers"]["work"].update(session_identity=[]),
            lambda d: d["providers"].update(
                gpt={"adapter": "claude", "model": "x", "effort": "high"}
            ),
            lambda d: d["providers"].update(
                claude={"adapter": "codex", "model": "x", "effort": "high"}
            ),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                document = json.loads(json.dumps(self.document))
                mutate(document)
                with self.assertRaises(ConfigurationError) as caught:
                    self.load(document)
                self.assertEqual(
                    str(caught.exception), "Invalid providers file configuration"
                )

    def test_auth_targets_sources_and_plaintext_are_rejected(self):
        cases = [
            ("work", {"ANTHROPIC_API_KEY": "sk-private-cleartext"}),
            (
                "work",
                {
                    "ANTHROPIC_API_KEY": "WORK_KEY",
                    "ANTHROPIC_AUTH_TOKEN": "OTHER_TOKEN",
                },
            ),
            ("work", {"ANTHROPIC_BASE_URL": "WORK_URL"}),
            ("code", {"PATH": "CODE_KEY"}),
            ("code", {"LD_SECRET": "CODE_KEY"}),
        ]
        cases += [
            ("code", {"CUSTOM_API_KEY": source})
            for source in (
                "PATH",
                "HOME",
                "USER",
                "SHELL",
                "TMPDIR",
                "PYTHONPATH",
                "LD_PRELOAD",
                "DOCKER_HOST",
                "UV_CACHE_DIR",
                "${CODE_KEY}",
            )
        ]
        for provider, references in cases:
            with self.subTest(provider=provider, references=references):
                document = json.loads(json.dumps(self.document))
                document["providers"][provider]["environment"] = references
                with self.assertRaises(ConfigurationError):
                    self.load(document)

    def test_missing_empty_relative_and_nonregular_files_fail_without_writes(self):
        for path in (self.root / "absent", self.root, Path("relative.yaml"), Path("")):
            with self.subTest(path=path), self.assertRaises(ConfigurationError):
                load_provider_config(path, environment={})
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaises(ConfigurationError):
            load_provider_config(fifo, environment={})
        self.path.write_bytes(b"x" * (1024 * 1024 + 1))
        with self.assertRaises(ConfigurationError):
            load_provider_config(self.path, environment={})
        with self.assertRaises(ConfigurationError):
            load_settings(
                repo_root=self.root, environment={"VIBESIM_AGENT_PROVIDERS_FILE": ""}
            )


if __name__ == "__main__":
    unittest.main()
