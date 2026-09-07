import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "with_claude_env.py"
spec = importlib.util.spec_from_file_location("claude_startup", SCRIPT)
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


class ClaudeStartupTests(unittest.TestCase):
    def discover(self, text, environment=None):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".bashrc").write_text(text)
            return startup.discover(environment or {}, Path(tmp))

    def test_function_and_alias(self):
        for definition in (
            'claude() {\n ANTHROPIC_BASE_URL="https://example.test" \\\n ANTHROPIC_AUTH_TOKEN="test secret" \\\n command claude "$@"\n}',
            "alias claude='ANTHROPIC_BASE_URL=https://example.test ANTHROPIC_AUTH_TOKEN=\"test secret\" claude'",
        ):
            with self.subTest(definition=definition):
                result = self.discover(definition)
                self.assertEqual(result["ANTHROPIC_AUTH_TOKEN"], "test secret")
                self.assertEqual(result["ANTHROPIC_BASE_URL"], "https://example.test")

    def test_explicit_auth_does_not_inherit_another_provider(self):
        explicit = {"ANTHROPIC_API_KEY": "explicit"}
        self.assertEqual(
            self.discover(
                "claude() { ANTHROPIC_AUTH_TOKEN=other ANTHROPIC_BASE_URL=https://other.test claude; }",
                explicit,
            ),
            explicit,
        )

    def test_existing_url_cannot_pick_up_different_provider_token(self):
        explicit = {"ANTHROPIC_BASE_URL": "https://explicit.test"}
        self.assertEqual(
            self.discover(
                "claude() { ANTHROPIC_AUTH_TOKEN=other ANTHROPIC_BASE_URL=https://other.test claude }",
                explicit,
            ),
            explicit,
        )

    def test_unrelated_function_ignored_and_variables_supported(self):
        result = self.discover(
            "claudek() { ANTHROPIC_AUTH_TOKEN=wrong claude }\n"
            'claude() { ANTHROPIC_AUTH_TOKEN="${MY_TOKEN}" command claude "$@" }',
            {"MY_TOKEN": "correct"},
        )
        self.assertEqual(result["ANTHROPIC_AUTH_TOKEN"], "correct")
        self.assertNotIn(
            "ANTHROPIC_AUTH_TOKEN",
            self.discover('claude() { ANTHROPIC_AUTH_TOKEN="$MISSING" claude }'),
        )

    def test_dynamic_code_and_other_commands_are_never_executed(self):
        with TemporaryDirectory() as tmp:
            marker = Path(tmp) / "executed"
            for body in (
                f'ANTHROPIC_AUTH_TOKEN="$(touch {marker})" claude',
                f"ANTHROPIC_AUTH_TOKEN=token claude; touch {marker}",
                "ANTHROPIC_AUTH_TOKEN=token echo claude",
            ):
                self.assertNotIn(
                    "ANTHROPIC_AUTH_TOKEN", self.discover(f"claude() {{ {body} }}")
                )
                self.assertFalse(marker.exists())

    def test_launcher_passes_credentials_without_printing_them(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / ".bashrc").write_text(
                'claude() { ANTHROPIC_AUTH_TOKEN=private-test-value command claude "$@" }'
            )
            env = {k: v for k, v in os.environ.items() if k not in startup.ALLOWED}
            env.update(HOME=tmp, CLAUDE_DISCOVER_SHELL_ENV="1")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    sys.executable,
                    "-c",
                    'import os; assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "private-test-value"',
                ],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout + result.stderr, "")
            env["CLAUDE_DISCOVER_SHELL_ENV"] = "0"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    sys.executable,
                    "-c",
                    'import os; assert "ANTHROPIC_AUTH_TOKEN" not in os.environ',
                ],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
