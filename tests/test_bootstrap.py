import asyncio
import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from vibesim_agent import bootstrap
from vibesim_agent.storage.database import Database


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.repo = self.root / "agent"
        self.repo.mkdir()
        self.main = self.root / "main"
        self.main.mkdir()
        self.state = self.root / "state"
        self.home = self.root / "home"
        self.home.mkdir()
        self.environment = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.home),
            "VIBESIM_AGENT_MAIN_DIR": str(self.main),
            "VIBESIM_AGENT_WORKSPACES_ROOT": str(self.state),
        }
        self.git("init", "--template=")
        self.git("config", "user.name", "Bootstrap test")
        self.git("config", "user.email", "bootstrap@example.invalid")
        (self.main / "AGENTS.md").write_text("main instructions")
        (self.main / "uv.lock").write_text("lock")
        (self.main / "logs").mkdir()
        (self.main / "logs/result").write_text("preserved result")
        self.git("add", "AGENTS.md", "uv.lock")
        self.git("-c", "commit.gpgsign=false", "commit", "-m", "initial")
        self.mcp = self.repo / "vibesim_agent/analyzer_evidence_mcp"
        self.mcp.mkdir(parents=True)
        (self.mcp / "server.py").write_text("# test MCP entrypoint\n")

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.main), *args],
            env=self.environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()

    def initialize(self):
        return bootstrap.initialize_state(
            environment=self.environment, repo_root=self.repo
        )

    def create(self):
        application = bootstrap.create_application(
            environment=self.environment, repo_root=self.repo
        )
        if hasattr(application.state, "recovery"):
            self.addCleanup(application.state.recovery.close)
        return application

    @staticmethod
    def assembled_stub(settings, **options):
        options["ownership"].close()
        return SimpleNamespace(state=SimpleNamespace())

    def source_bytes(self):
        return {
            str(path.relative_to(self.main)): path.read_bytes()
            for path in self.main.rglob("*")
            if path.is_file()
        }

    def test_initialize_explicit_main_database_index_without_touching_source(self):
        before = self.source_bytes()
        with patch("subprocess.run", wraps=subprocess.run) as run:
            descriptor = self.initialize()
        self.assertTrue(all(call.args[0][0] == "git" for call in run.call_args_list))
        self.assertEqual(descriptor["workspace_id"], "w_main")
        self.assertEqual(descriptor["storage_kind"], "external")
        self.assertEqual(descriptor["repo_path"], str(self.main))
        self.assertEqual(descriptor["logs_path"], str(self.main / "logs"))
        self.assertIsNone(descriptor["base_workspace_id"])
        self.assertEqual(descriptor["base_revision"], self.git("rev-parse", "HEAD"))
        self.assertEqual(
            json.loads((self.state / "w_main/workspace.json").read_text()), descriptor
        )
        self.assertTrue((self.state / "registry.json").is_file())
        with Database(self.state / "w_main/workspace.sqlite").connect() as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
                0,
            )
        self.assertEqual(self.source_bytes(), before)

    def test_existing_root_even_empty_is_never_adopted_or_overwritten(self):
        self.state.mkdir()
        with self.assertRaises(FileExistsError):
            self.initialize()
        self.assertEqual(list(self.state.iterdir()), [])
        marker = self.state / "someone-elses-file"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.initialize()
        self.assertEqual(marker.read_text(), "keep")

    def test_failed_initialize_rolls_back_only_owned_state_and_allows_retry(self):
        before = self.source_bytes()
        with (
            patch.object(
                Database, "create", side_effect=OSError("injected database failure")
            ),
            self.assertRaises(OSError),
        ):
            self.initialize()
        self.assertFalse(self.state.exists())
        self.assertEqual(self.source_bytes(), before)
        self.assertEqual(self.initialize()["workspace_id"], "w_main")

    def test_configuration_uses_snapshot_home_and_allows_unrelated_codex_home(self):
        explicit = dict(self.environment, CODEX_HOME="/unrelated/cli/home")
        with patch.dict(os.environ, {"HOME": "/not-the-snapshot"}, clear=True):
            settings = bootstrap.configuration(
                environment=explicit, repo_root=self.repo
            )
        self.assertEqual(settings.providers["gpt"].home, self.home / ".codex")
        self.assertEqual(settings.agent.workspaces_root, self.state)
        self.assertFalse(self.state.exists())

    def test_factory_assembly_uses_existing_gitlinks_and_stable_namespace(self):
        self.initialize()
        commit = self.git("rev-parse", "HEAD")
        for path in ("vendor/present", "vendor/missing"):
            self.git("update-index", "--add", "--cacheinfo", f"160000,{commit},{path}")
        (self.main / "vendor/present").mkdir(parents=True)
        with patch.object(
            bootstrap, "build_application", side_effect=self.assembled_stub
        ) as build:
            self.create()
            (settings,) = build.call_args.args
            first = build.call_args.kwargs
            self.create()
            second = build.call_args.kwargs
        self.assertEqual(settings.agent.repo_root, self.repo)
        self.assertEqual(first["prompts_directory"], self.state / ".prompts")
        self.assertEqual(first["mcp_directory"], self.mcp)
        self.assertEqual(first["managed_context"], "/opt/vibesim/managed/context.json")
        self.assertEqual(first["submodules"], (Path("vendor/present"),))
        self.assertEqual(first["namespace"], second["namespace"])
        self.assertTrue(first["namespace"])
        self.assertTrue(callable(first["providers"]))

    def test_real_factory_has_settings_and_does_not_start_runtime(self):
        self.initialize()
        application = self.create()
        self.assertEqual(application.state.settings.agent.workspaces_root, self.state)
        self.assertFalse((self.state / "w_main/runtime").exists())
        self.assertEqual(application.state.turns.storage("w_main").turns.running(), [])
        self.assertTrue((self.state / ".prompts").is_dir())

    def test_archived_legacy_database_preflight_leaves_prompt_directory_absent(self):
        self.initialize()
        archived = self.state / "w_old"
        archived.mkdir()
        descriptor = json.loads((self.state / "w_main/workspace.json").read_text())
        descriptor.update(workspace_id="w_old", state="archived")
        (archived / "workspace.json").write_text(json.dumps(descriptor))
        database = archived / "workspace.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE old(value TEXT)")
        before = database.read_bytes()
        with (
            patch.object(bootstrap, "build_application") as build,
            self.assertRaises(ValueError),
        ):
            self.create()
        build.assert_not_called()
        self.assertEqual(database.read_bytes(), before)
        self.assertFalse((self.state / ".prompts").exists())

    def test_cli_serve_forwards_single_worker_and_rejects_extra_process_flags(self):
        from vibesim_agent import __main__ as cli

        self.initialize()
        application = self.create()
        with (
            patch.object(cli, "create_application", return_value=application),
            patch("uvicorn.run") as run,
        ):
            cli.main(["serve"])
            run.assert_called_once_with(
                application,
                host=application.state.settings.agent.bind,
                port=application.state.settings.agent.port,
                workers=1,
                lifespan="on",
            )
            run.reset_mock()
            for flag in ("--reload", "--workers"):
                with (
                    self.subTest(flag=flag),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    cli.main(["serve", flag])
            run.assert_not_called()

    def test_retired_keys_rejected_without_values_and_reference_remains_available(self):
        from vibesim_agent import __main__ as cli

        for key in ("OPENROUTE_KEY", "CODEX_MODEL", "VIBESIM_WORKSPACES_ROOT"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as caught:
                    bootstrap.configuration(
                        environment={**self.environment, key: "private-value"},
                        repo_root=self.repo,
                    )
                self.assertIn(key, str(caught.exception))
                self.assertNotIn("private-value", str(caught.exception))
        output = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {**self.environment, "OPENROUTE_KEY": "private-value"},
                clear=True,
            ),
            contextlib.redirect_stdout(output),
        ):
            cli.main(["env-reference"])
        self.assertIn("VIBESIM_AGENT_WORKSPACES_ROOT", output.getvalue())
        self.assertNotIn("private-value", output.getvalue())
        self.assertFalse(self.state.exists())

    def test_noargument_factory_matches_explicit_environment_snapshot(self):
        self.initialize()
        with (
            patch.dict(os.environ, self.environment, clear=True),
            patch.object(
                bootstrap, "build_application", side_effect=self.assembled_stub
            ) as build,
        ):
            bootstrap.create_application()
            implicit_settings = build.call_args.args[0]
            implicit_options = build.call_args.kwargs
            bootstrap.create_application(environment=dict(self.environment))
            explicit_settings = build.call_args.args[0]
            explicit_options = build.call_args.kwargs
        self.assertEqual(implicit_settings, explicit_settings)
        for key in (
            "namespace",
            "prompts_directory",
            "mcp_directory",
            "submodules",
            "workspace_environment",
        ):
            self.assertEqual(implicit_options[key], explicit_options[key])

    def test_missing_prerequisites_fail_before_prompt_rendering(self):
        with self.assertRaises(ValueError):
            self.create()
        self.assertFalse(self.state.exists())
        self.initialize()
        for missing in (
            self.main / "AGENTS.md",
            self.main / "uv.lock",
            self.mcp / "server.py",
        ):
            original = missing.read_bytes()
            missing.unlink()
            try:
                with (
                    self.assertRaises(ValueError),
                    patch.object(bootstrap, "build_application") as build,
                ):
                    self.create()
                build.assert_not_called()
                self.assertFalse((self.state / ".prompts").exists())
            finally:
                missing.write_bytes(original)

    def test_import_does_not_load_environment_or_launch_runtime(self):
        script = """
import os
import sys
from pathlib import Path
from unittest.mock import patch
# Preload framework dependencies so the guard targets Agent import-time actions.
import vibesim_agent.application
import vibesim_agent.composition
import uvicorn
original = type(os.environ).__getitem__
def guarded(environment, key):
    frame = sys._getframe(1)
    while frame and frame.f_globals.get('__name__') in {'os', 'collections.abc', '_collections_abc'}:
        frame = frame.f_back
    if frame and frame.f_globals.get('__name__', '').startswith('vibesim_agent'):
        raise AssertionError('Agent read environment: ' + key)
    return original(environment, key)
with patch.object(type(os.environ), '__getitem__', new=guarded), \\
     patch.object(Path, 'mkdir', side_effect=AssertionError('state write')), \\
     patch('subprocess.Popen', side_effect=AssertionError('process start')), \\
     patch('sqlite3.connect', side_effect=AssertionError('database access')):
    import vibesim_agent.bootstrap
    import vibesim_agent.__main__
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_second_factory_cannot_rewrite_prompts_owned_by_live_backend(self):
        self.initialize()

        async def check():
            first = self.create()
            async with first.router.lifespan_context(first):
                prompt = self.state / ".prompts/assistant.txt"
                prompt.write_text("live backend prompt revision")
                before = {
                    path.name: path.read_bytes()
                    for path in prompt.parent.iterdir()
                    if path.is_file()
                }
                with self.assertRaisesRegex(RuntimeError, "owned by another backend"):
                    self.create()
                self.assertEqual(
                    {
                        path.name: path.read_bytes()
                        for path in prompt.parent.iterdir()
                        if path.is_file()
                    },
                    before,
                )

        asyncio.run(check())

    def test_prompt_directory_symlink_rejected_without_touching_external_files(self):
        self.initialize()
        outside = self.root / "outside-prompts"
        outside.mkdir()
        marker = outside / "assistant.txt"
        marker.write_text("external prompt")
        (self.state / ".prompts").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.create()
        self.assertEqual(list(outside.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "external prompt")

    def test_dangling_state_symlink_is_not_followed_by_initialization(self):
        target = self.root / "not-created"
        self.state.symlink_to(target, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.initialize()
        self.assertTrue(self.state.is_symlink())
        self.assertFalse(target.exists())

    def test_existing_prompt_file_symlink_rejected_before_external_overwrite(self):
        self.initialize()
        prompts = self.state / ".prompts"
        prompts.mkdir()
        outside = self.root / "private-file"
        outside.write_text("preserve")
        (prompts / "assistant.txt").symlink_to(outside)
        with self.assertRaises(ValueError):
            self.create()
        self.assertEqual(outside.read_text(), "preserve")

    def test_cli_init_outputs_descriptor_and_existing_state_is_clear_error(self):
        from vibesim_agent import __main__ as cli

        output = io.StringIO()
        with (
            patch.dict(os.environ, self.environment, clear=True),
            contextlib.redirect_stdout(output),
        ):
            cli.main(["init"])
        self.assertEqual(json.loads(output.getvalue())["workspace_id"], "w_main")
        error = io.StringIO()
        with (
            patch.dict(os.environ, self.environment, clear=True),
            contextlib.redirect_stderr(error),
            self.assertRaises(SystemExit) as caught,
        ):
            cli.main(["init"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("already exists", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())
