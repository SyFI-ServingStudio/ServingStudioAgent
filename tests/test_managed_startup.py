import copy
import io
import json
import os
import unittest
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests import test_migrate_workspaces_v1 as fixtures
from tools.migrate_v1_database import MigrationError
from tools.migration_files import inventory_tree
from tools.startup_selection import _root
from vibesim_agent import __main__ as cli
from vibesim_agent import startup
from vibesim_agent.bootstrap import configuration as real_configuration
from vibesim_agent.providers.builtin import session_scope as real_session_scope
from vibesim_agent.settings import ConnectionSettings
from vibesim_agent.storage.database import Database


class ManagedStartupTests(unittest.TestCase):
    run_migration = fixtures.WorkspaceMigrationTests.run_migration

    def setUp(self):
        fixtures.WorkspaceMigrationTests.setUp(self)
        identity = self.families["old-family"]
        self.models = {"old-model": identity}
        self.environment = {"DOCKER_HOST": "unix:///var/run/docker.sock"}
        self.config_path = self.root / "startup.json"
        self.mapping_path = self.root / "mapping.json"
        self.deployment_path = self.root / "deployment.json"
        self.selection = self.root / "selection.json"
        self.receipt = self.root / "receipt.json"
        self.config = {
            "format": 1,
            "source": str(self.source),
            "target": str(self.target),
            "selection": str(self.selection),
            "receipt": str(self.receipt),
            "deployment": str(self.deployment_path),
            "mapping": str(self.mapping_path),
            "mode": "production",
        }
        self.mapping = {
            "models": {
                "old-model": {
                    "provider_id": identity.provider_id,
                    "session_scope": identity.session_scope,
                }
            },
            "families": {
                "old-family": {
                    "provider_id": identity.provider_id,
                    "session_scope": identity.session_scope,
                }
            },
            "runners": self.runners,
        }
        self.write_inputs()
        self.deployment_path.write_text(json.dumps({"source": _root(self.source)}))
        self.quiet = False
        self.events = []
        self.settings = SimpleNamespace(
            providers={identity.provider_id: object()},
            connections={identity.provider_id: ConnectionSettings(adapter="codex")},
            agent=SimpleNamespace(main_dir=self.root / "main"),
        )
        self.enterContext(
            patch.object(startup, "configuration", return_value=self.settings)
        )
        self.source_check = self.enterContext(patch.object(startup, "_source"))
        self.scope = self.enterContext(
            patch.object(startup, "session_scope", return_value=identity.session_scope)
        )
        self.deployment = self.enterContext(
            patch.object(
                startup,
                "TmuxDeployment",
                return_value=SimpleNamespace(quiesce=self.quiesce),
            )
        )
        self.before = inventory_tree(self.source)

    def write_inputs(self):
        self.config_path.write_text(json.dumps(self.config))
        self.mapping_path.write_text(json.dumps(self.mapping))

    @contextmanager
    def quiesce(self, source):
        self.assertEqual(source, self.source)
        self.assertFalse(self.quiet)
        self.quiet = True
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")
            self.quiet = False

    def managed(self):
        return startup.ManagedStartup(self.config_path, environment=self.environment)

    def assert_no_shutdown_or_target(self):
        self.deployment.assert_not_called()
        self.assertFalse(self.target.exists())
        self.assertFalse(self.selection.exists())
        self.assertEqual(inventory_tree(self.source), self.before)

    def test_two_preparations_migrate_once_and_preserve_new_messages(self):
        original_environment = dict(self.environment)
        with self.managed().prepare() as effective:
            self.assertTrue(self.quiet)
            self.assertEqual(
                effective["VIBESIM_AGENT_WORKSPACES_ROOT"], str(self.target)
            )
            with Database(self.target / "w_main/workspace.sqlite").connect(
                write=True
            ) as connection:
                connection.execute(
                    "INSERT INTO messages(conversation_id,role,content,ts,metadata_json,turn_id) VALUES ('c','assistant','after migration',10,'{}','t')"
                )
        record = self.selection.read_bytes()
        with (
            patch(
                "tools.startup_migration.migrate_workspaces",
                side_effect=AssertionError("must not recopy"),
            ),
            self.managed().prepare(),
        ):
            self.assertTrue(self.quiet)
            with Database(
                self.target / "w_main/workspace.sqlite"
            ).connect() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM messages WHERE content='after migration'"
                    ).fetchone()[0],
                    1,
                )
        self.assertEqual(self.events, ["enter", "exit", "enter", "exit"])
        self.assertEqual(self.selection.read_bytes(), record)
        self.assertEqual(self.environment, original_environment)
        self.assertEqual(inventory_tree(self.source), self.before)
        self.assertEqual(self.deployment.call_count, 2)
        self.assertEqual(self.deployment.call_args.kwargs["receipt"], self.receipt)
        self.assertEqual(self.deployment.call_args.kwargs["target"], self.target)

    def test_selected_root_is_read_only_and_legacy_without_record_refuses(self):
        with self.assertRaisesRegex(MigrationError, "not published"):
            self.managed().selected_root()
        self.assert_no_shutdown_or_target()
        with self.managed().prepare():
            pass
        self.deployment.reset_mock()
        target_before = inventory_tree(self.target)
        record = self.selection.read_bytes()
        with (
            patch.object(
                cli,
                "ManagedStartup",
                side_effect=lambda path, environment: startup.ManagedStartup(
                    path, environment=environment
                ),
            ),
            patch.dict(os.environ, self.environment, clear=True),
            redirect_stdout(io.StringIO()) as output,
        ):
            cli.main(["selected-root", "--startup-config", str(self.config_path)])
        self.assertEqual(output.getvalue().strip(), str(self.target))
        self.deployment.assert_not_called()
        self.assertEqual(inventory_tree(self.target), target_before)
        self.assertEqual(self.selection.read_bytes(), record)
        self.assertEqual(inventory_tree(self.source), self.before)

    def test_scope_change_during_migration_keeps_target_but_refuses_application(self):
        from tools import startup_migration

        migrate = startup_migration.migrate_workspaces

        def migrate_then_change(*args, **kwargs):
            report = migrate(*args, **kwargs)
            self.scope.return_value = "changed-during-copy"
            return report

        owner = self.managed()
        with (
            patch.object(
                startup_migration, "migrate_workspaces", side_effect=migrate_then_change
            ),
            self.assertRaisesRegex(MigrationError, "scope differs"),
            owner.prepare(),
        ):
            self.fail("changed profile reached application startup")
        self.assertEqual(self.events, ["enter", "exit"])
        self.assertTrue(self.target.is_dir())
        self.assertTrue(self.selection.is_file())
        self.assertEqual(inventory_tree(self.source), self.before)
        selected = self.selection.read_bytes()
        self.deployment.reset_mock()
        with self.assertRaisesRegex(MigrationError, "scope differs"):
            self.managed()
        self.deployment.assert_not_called()
        self.scope.return_value = "session:scope"
        with (
            patch.object(
                startup_migration,
                "migrate_workspaces",
                side_effect=AssertionError("must not recopy the accepted target"),
            ),
            self.managed().prepare() as effective,
        ):
            self.assertEqual(
                effective["VIBESIM_AGENT_WORKSPACES_ROOT"], str(self.target)
            )
        self.assertEqual(self.selection.read_bytes(), selected)

    def test_corrupt_selection_never_falls_back_or_stops_services(self):
        self.selection.write_bytes(b"not json")
        for method in ("selected_root", "prepare"):
            with self.subTest(method=method), self.assertRaises(MigrationError):
                if method == "prepare":
                    with self.managed().prepare():
                        self.fail("corrupt selection yielded")
                else:
                    self.managed().selected_root()
        self.deployment.assert_not_called()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.selection.read_bytes(), b"not json")
        self.assertEqual(inventory_tree(self.source), self.before)

    def test_scope_runner_and_absent_provider_fail_before_shutdown(self):
        original = copy.deepcopy(self.mapping)
        for case in ("scope", "runner", "provider"):
            self.mapping = copy.deepcopy(original)
            self.scope.return_value = "session:scope"
            self.settings.providers = {"session_provider": object()}
            if case == "scope":
                self.scope.return_value = "different-scope"
            elif case == "runner":
                self.mapping["runners"] = {"old-family": "claude"}
            else:
                self.settings.providers = {}
            self.write_inputs()
            with self.subTest(case=case), self.assertRaises(MigrationError):
                self.managed()
            self.assert_no_shutdown_or_target()

    def test_named_claude_connection_dispatches_scope_validation_to_claude(self):
        owner = self.managed()
        identity = replace(self.families["old-family"], provider_id="claude_work")
        owner.options["models"] = {"old-model": identity}
        owner.options["families"] = {"old-family": identity}
        owner.options["runners"] = {"old-family": "claude"}
        self.settings.providers = {"claude_work": object()}
        self.settings.connections = {
            "claude_work": ConnectionSettings(adapter="claude")
        }
        self.scope.reset_mock()
        owner._validate_runtime()
        self.assertEqual(self.scope.call_count, 2)
        self.assertTrue(
            all(
                call.args == (self.settings, "claude_work", "claude")
                for call in self.scope.call_args_list
            )
        )
        self.assert_no_shutdown_or_target()
        owner.options["runners"] = {"old-family": "codex"}
        with self.assertRaisesRegex(MigrationError, "runner differs"):
            owner._validate_runtime()
        self.assert_no_shutdown_or_target()

    def test_invalid_configuration_paths_and_identity_fail_before_shutdown(self):
        original = copy.deepcopy(self.config)
        for case in (
            "collision",
            "source-path",
            "external-path",
            "relative",
            "mode",
            "format",
            "extra",
            "deployment-source",
        ):
            self.config = copy.deepcopy(original)
            if case == "collision":
                self.config["receipt"] = self.config["selection"]
            elif case == "source-path":
                self.config["receipt"] = str(self.source / "receipt.json")
            elif case == "external-path":
                self.config["receipt"] = str(self.root / "main" / "receipt.json")
            elif case == "relative":
                self.config["target"] = "relative"
            elif case == "mode":
                self.config["mode"] = "unknown"
            elif case == "format":
                self.config["format"] = True
            elif case == "extra":
                self.config["unknown"] = 1
            else:
                self.deployment_path.write_text(json.dumps({"source": {}}))
            self.write_inputs()
            with self.subTest(case=case), self.assertRaises(MigrationError):
                self.managed()
            self.assert_no_shutdown_or_target()

    def test_current_fast_path_needs_no_deployment_access_or_record(self):
        self.run_migration()
        current = self.target
        self.config["source"] = str(current)
        self.config["target"] = str(self.root / "unused-target")
        self.deployment_path.write_text(json.dumps({"source": _root(current)}))
        self.write_inputs()
        before = inventory_tree(current)
        owner = self.managed()
        self.assertEqual(owner.selected_root(), current)
        with owner.prepare() as effective:
            self.assertEqual(effective["VIBESIM_AGENT_WORKSPACES_ROOT"], str(current))
        self.deployment.assert_not_called()
        self.assertFalse(self.selection.exists())
        self.assertEqual(inventory_tree(current), before)

    def test_missing_database_model_or_session_mapping_refuses_before_shutdown(self):
        original = copy.deepcopy(self.mapping)
        for field in ("models", "families"):
            self.mapping = copy.deepcopy(original)
            self.mapping[field] = {}
            if field == "families":
                self.mapping["runners"] = {}
            self.write_inputs()
            with self.subTest(field=field), self.assertRaises(MigrationError):
                self.managed()
            self.assert_no_shutdown_or_target()

    def test_real_profile_scope_is_checked_before_shutdown_and_detects_backend_change(
        self,
    ):
        home = self.root / "profile"
        home.mkdir()
        profile = home / "config.toml"
        profile.write_text(
            'model_provider="fixture"\n[model_providers.fixture]\nbase_url="http://first.invalid/v1"\nwire_api="responses"\n'
        )
        self.environment.update(
            {
                "HOME": str(self.root),
                "VIBESIM_AGENT_MAIN_DIR": str(self.root / "main"),
                "VIBESIM_PROVIDER_GPT_HOME": str(home),
            }
        )
        settings = real_configuration(
            environment=self.environment
            | {"VIBESIM_AGENT_WORKSPACES_ROOT": str(self.target)}
        )
        scope = real_session_scope(settings, "gpt", "codex")
        identity = {"provider_id": "gpt", "session_scope": scope}
        self.mapping["models"] = {"old-model": identity}
        self.mapping["families"] = {"old-family": identity}
        self.write_inputs()
        with (
            patch.object(startup, "configuration", side_effect=real_configuration),
            patch.object(startup, "session_scope", side_effect=real_session_scope),
        ):
            self.managed()
            self.assert_no_shutdown_or_target()
            profile.write_text(
                profile.read_text().replace("first.invalid", "second.invalid")
            )
            self.assertNotEqual(real_session_scope(settings, "gpt", "codex"), scope)
            with self.assertRaisesRegex(MigrationError, "scope differs"):
                self.managed()
        self.assert_no_shutdown_or_target()

    def test_model_without_any_historical_sessions_migrates_with_empty_families(self):
        self.writer.execute("DELETE FROM codex_sessions")
        self.writer.commit()
        self.before = inventory_tree(self.source)
        self.mapping["families"] = {}
        self.mapping["runners"] = {}
        self.write_inputs()
        with self.managed().prepare():
            pass
        with Database(self.target / "w_main/workspace.sqlite").connect() as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM agent_sessions").fetchone()[0],
                0,
            )
        self.assertEqual(inventory_tree(self.source), self.before)

    def test_malformed_mapping_values_reject_cleanly_without_shutdown(self):
        original = copy.deepcopy(self.mapping)
        for case in ("mode", "runner", "external", "nested-external"):
            self.mapping = copy.deepcopy(original)
            self.config["mode"] = "production"
            if case == "mode":
                self.config["mode"] = []
            elif case == "runner":
                self.mapping["runners"] = {"old-family": []}
            elif case == "external":
                self.mapping["external_paths"] = []
            else:
                self.mapping["external_paths"] = {"w_main": {"repo_path": []}}
            self.write_inputs()
            with self.subTest(case=case), self.assertRaises(MigrationError):
                self.managed()
            self.assert_no_shutdown_or_target()

    def test_main_checkout_and_external_target_overlap_refuse_before_shutdown(self):
        self.settings.agent.main_dir = self.root / "different-main"
        with self.assertRaisesRegex(MigrationError, "w_main differs"):
            self.managed()
        self.assert_no_shutdown_or_target()
        self.settings.agent.main_dir = self.root / "main"
        for target in (self.root / "main" / "state", self.root):
            self.config["target"] = str(target)
            self.write_inputs()
            with self.subTest(target=target), self.assertRaises(MigrationError):
                self.managed()
            self.assert_no_shutdown_or_target()

    def test_selected_root_rejects_changed_external_paths_without_shutdown(self):
        first = self.root / "first-copy"
        second = self.root / "second-copy"
        for repo in (first, second):
            (repo / "logs").mkdir(parents=True)
        self.config["mode"] = "rehearsal"

        def use(repo):
            self.settings.agent.main_dir = repo
            self.mapping["external_paths"] = {
                "w_main": {"repo_path": str(repo), "logs_path": str(repo / "logs")}
            }
            self.write_inputs()

        use(first)
        with self.managed().prepare():
            pass
        self.deployment.reset_mock()
        record = self.selection.read_bytes()
        target_before = inventory_tree(self.target)
        use(second)
        with self.assertRaisesRegex(MigrationError, "external paths changed"):
            self.managed().selected_root()
        self.deployment.assert_not_called()
        self.assertEqual(self.selection.read_bytes(), record)
        self.assertEqual(inventory_tree(self.target), target_before)
        self.assertEqual(inventory_tree(self.source), self.before)

    def test_cli_context_covers_factory_run_and_recovery_close_including_failures(self):
        for failure in (None, "factory", "run", "close"):
            self.events.clear()
            holder = {}

            def factory(*, environment, failure=failure, holder=holder):
                self.assertTrue(self.quiet)
                self.assertEqual(
                    environment["VIBESIM_AGENT_WORKSPACES_ROOT"], str(self.target)
                )
                self.events.append("factory")
                if failure == "factory":
                    raise RuntimeError("factory failed")
                return holder["app"]

            def run(*args, failure=failure, **kwargs):
                self.assertTrue(self.quiet)
                self.events.append("run")
                self.assertEqual(
                    kwargs,
                    {
                        "host": "127.0.0.1",
                        "port": 12345,
                        "workers": 1,
                        "lifespan": "on",
                    },
                )
                if failure == "run":
                    raise RuntimeError("run failed")

            def close(failure=failure):
                self.assertTrue(self.quiet)
                self.events.append("close")
                if failure == "close":
                    raise RuntimeError("close failed")

            app = SimpleNamespace(
                state=SimpleNamespace(
                    settings=SimpleNamespace(
                        agent=SimpleNamespace(bind="127.0.0.1", port=12345)
                    ),
                    recovery=SimpleNamespace(close=Mock(side_effect=close)),
                )
            )
            holder["app"] = app
            with (
                self.subTest(failure=failure),
                patch.dict(os.environ, self.environment, clear=True),
                patch.object(cli, "create_application", side_effect=factory),
                patch.object(cli.uvicorn, "run", side_effect=run),
            ):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, failure + " failed"):
                        cli.main(["serve", "--startup-config", str(self.config_path)])
                else:
                    cli.main(["serve", "--startup-config", str(self.config_path)])
            expected = (
                ["enter", "factory"]
                + ([] if failure == "factory" else ["run", "close"])
                + ["exit"]
            )
            self.assertEqual(self.events, expected)
            self.assertFalse(self.quiet)
            self.assertTrue(self.selection.exists())
            self.assertEqual(inventory_tree(self.source), self.before)
