import contextlib
import hashlib
import io
import json
import shutil
import sqlite3
import unittest
from unittest.mock import patch

from tests import test_migrate_database_v1 as fixtures
from tests.legacy_workspaces import create_legacy_workspace
from tools import migrate_v1_workspaces as migration
from tools.migrate_v1_database import ProviderIdentity
from tools.migration_files import inventory_tree
from vibesim_agent.storage.database import Database


class WorkspaceMigrationTests(unittest.TestCase):
    def setUp(self):
        fixtures.DatabaseMigrationTests.setUp(self)
        self.source_database = self.source
        self.source = self.source.parent.parent
        self.target = self.root / "migrated"
        self.main = self.root / "main"
        (self.main / "logs").mkdir()
        (self.main / "logs/result").write_bytes(b"external result")
        create_legacy_workspace(
            self.source, self.main, workspace_id="w_managed", name="Managed"
        )
        managed = self.source / "w_managed/repo"
        (managed / ".git").mkdir(parents=True)
        (managed / ".git/index").write_bytes(b"preserved git index")
        (managed / "dirty").write_bytes(b"dirty working file")
        (managed / "logs").mkdir()
        (managed / "logs/output").write_bytes(b"managed result")
        (self.source / "unknown-root-file").write_bytes(b"unknown retained")
        (self.source / "w_main/jobs").mkdir()
        (self.source / "w_main/jobs/job.json").write_bytes(b"job raw bytes")
        self.legacy = self.source / "w_main/codex/c"
        self.role = self.legacy / "assistant"
        (self.role / "sessions").mkdir(parents=True)
        (self.role / "sessions/local.jsonl").write_bytes(b"local rollout")
        (self.role / "config.toml").write_text('model_provider="old"\n')
        (self.legacy / "sessions").mkdir()
        (self.legacy / "sessions/shared.jsonl").write_bytes(b"shared rollout")
        (self.legacy / "shell_snapshots").mkdir()
        (self.legacy / "shell_snapshots/environment.sh").write_bytes(b"shell state")
        self.runners = {"old-family": "codex"}

    def run_migration(self, **kwargs):
        options = {
            "models": self.models,
            "families": self.families,
            "runners": self.runners,
            "dry_run": False,
            "source_quiesced": True,
        }
        options.update(kwargs)
        return migration.migrate_workspaces(self.source, self.target, **options)

    def bytes_tree(self, root):
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }

    def active_home(self, scope="session:scope"):
        return (
            self.target
            / "w_main/runtime/c/assistant"
            / hashlib.sha256(scope.encode()).hexdigest()
        )

    def test_production_conversion_preserves_unknown_repo_jobs_old_home_and_archives(
        self,
    ):
        before = self.bytes_tree(self.source)
        report = self.run_migration()
        self.assertTrue(report["verified"])
        self.assertFalse(report["resume_verified"])
        self.assertFalse(report["external_paths_verified"])
        self.assertFalse(report["execution_isolation_verified"])
        self.assertEqual(
            json.loads((self.target / ".migration-v1/manifest.json").read_text()),
            report,
        )
        self.assertEqual(self.bytes_tree(self.source), before)
        for name in (
            "unknown-root-file",
            "w_main/jobs/job.json",
            "w_managed/repo/.git/index",
            "w_managed/repo/dirty",
            "w_managed/repo/logs/output",
        ):
            self.assertEqual((self.target / name).read_bytes(), before[name])
        self.assertEqual(
            self.bytes_tree(self.target / "w_main/codex"),
            self.bytes_tree(self.source / "w_main/codex"),
        )
        archive = self.target / ".migration-v1/original"
        for name in (
            "registry.json",
            "w_main/workspace.json",
            "w_main/workspace.sqlite",
            "w_main/workspace.sqlite-wal",
        ):
            self.assertEqual((archive / name).read_bytes(), before[name])
        descriptor = json.loads((self.target / "w_main/workspace.json").read_text())
        self.assertEqual(descriptor["repo_path"], str(self.main))
        self.assertEqual(descriptor["logs_path"], str(self.main / "logs"))
        for suffix in ("-wal", "-shm", "-journal"):
            self.assertFalse(
                (self.target / ("w_main/workspace.sqlite" + suffix)).exists()
            )
        index = json.loads((self.target / "registry.json").read_text())
        self.assertNotIn(".workspace-migrate-", json.dumps(index))
        logs = {
            row["workspace_id"]: (self.target / row["logs_root"]).resolve()
            for row in index["workspaces"]
        }
        self.assertEqual(
            logs,
            {
                "w_main": self.main / "logs",
                "w_managed": self.target / "w_managed/repo/logs",
            },
        )
        with Database(self.target / "w_main/workspace.sqlite").connect() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT session_scope FROM agent_sessions"
                ).fetchone()[0],
                "session:scope",
            )
            self.assertEqual(
                connection.execute(
                    "SELECT session_scope FROM role_settings LIMIT 1"
                ).fetchone()[0],
                "model:scope",
            )
        home = self.active_home()
        self.assertEqual((home / "sessions/local.jsonl").read_bytes(), b"local rollout")
        self.assertEqual(
            (home / "sessions/shared.jsonl").read_bytes(), b"shared rollout"
        )
        self.assertEqual(
            (home / "shell_snapshots/environment.sh").read_bytes(), b"shell state"
        )

    def test_claude_subdirectory_becomes_scope_root_without_codex_shared_merge(self):
        self.writer.execute("UPDATE codex_sessions SET family='claude-old'")
        self.writer.commit()
        claude = self.role / "claude"
        (claude / "projects").mkdir(parents=True)
        (claude / "projects/session.jsonl").write_bytes(b"claude resume")
        self.run_migration(
            families={"claude-old": ProviderIdentity("claude", "claude:scope")},
            runners={"claude-old": "claude"},
        )
        home = self.active_home("claude:scope")
        self.assertEqual(
            (home / "projects/session.jsonl").read_bytes(), b"claude resume"
        )
        self.assertFalse((home / "claude").exists())
        self.assertFalse((home / "sessions/shared.jsonl").exists())

    def test_dry_run_does_not_create_target_and_writing_requires_quiescence(self):
        before = self.bytes_tree(self.source)
        report = self.run_migration(dry_run=True, source_quiesced=False)
        self.assertFalse(report["verified"])
        self.assertFalse(report["resume_verified"])
        self.assertFalse(self.target.exists())
        with self.assertRaises(ValueError):
            self.run_migration(source_quiesced=False)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.bytes_tree(self.source), before)

    def test_conflicting_codex_shared_session_rejected_and_source_retained(self):
        collision = self.role / "sessions/shared.jsonl"
        collision.write_bytes(b"different local rollout")
        before = self.bytes_tree(self.source)
        with self.assertRaises(ValueError):
            self.run_migration()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.bytes_tree(self.source), before)
        collision.write_bytes(b"shared rollout")
        self.assertTrue(self.run_migration()["verified"])

    def test_import_marker_skips_shared_merge_but_preserves_legacy_tree(self):
        (self.role / ".legacy-shared-runtime-imported").write_text(
            "role-isolation-v1\n"
        )
        (self.role / "sessions/shared.jsonl").write_bytes(b"newer role rollout")
        self.run_migration()
        self.assertEqual(
            (self.active_home() / "sessions/shared.jsonl").read_bytes(),
            b"newer role rollout",
        )
        self.assertEqual(
            (self.target / "w_main/codex/c/sessions/shared.jsonl").read_bytes(),
            b"shared rollout",
        )

    def test_missing_home_unknown_runner_or_mapping_refuse_without_publication(self):
        for options in ({"families": {}}, {"runners": {}}, {"models": {}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_migration(**options)
            self.assertFalse(self.target.exists())
        shutil.rmtree(self.legacy)
        with self.assertRaises(ValueError):
            self.run_migration()
        self.assertFalse(self.target.exists())

    def test_rehearsal_requires_independent_external_repo_and_logs(self):
        with self.assertRaises(ValueError):
            self.run_migration(mode="rehearsal")
        independent = self.root / "rehearsal-repo"
        (independent / "logs").mkdir(parents=True)
        report = self.run_migration(
            mode="rehearsal",
            external_paths={
                "w_main": {"repo_path": independent, "logs_path": independent / "logs"}
            },
        )
        self.assertTrue(report["verified"])
        descriptor = json.loads((self.target / "w_main/workspace.json").read_text())
        self.assertEqual(descriptor["repo_path"], str(independent))
        self.assertEqual(descriptor["logs_path"], str(independent / "logs"))

    def test_rehearsal_alias_child_and_parent_of_old_external_root_are_rejected(self):
        alias = self.root / "main-alias"
        alias.symlink_to(self.main, target_is_directory=True)
        for root in (alias, self.main / "logs", self.main.parent):
            with self.subTest(root=root), self.assertRaises(ValueError):
                self.run_migration(
                    mode="rehearsal",
                    external_paths={"w_main": {"repo_path": root, "logs_path": root}},
                )
            self.assertFalse(self.target.exists())

    def test_existing_empty_target_and_dangling_target_are_not_overwritten(self):
        self.target.mkdir()
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_migration()
        self.assertEqual(list(self.target.iterdir()), [])
        self.target.rmdir()
        missing = self.root / "missing"
        self.target.symlink_to(missing, target_is_directory=True)
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_migration()
        self.assertTrue(self.target.is_symlink())
        self.assertFalse(missing.exists())

    def test_reserved_directory_and_descriptor_identity_mismatch_refuse(self):
        reserved = self.source / ".migration-v1"
        reserved.mkdir()
        with self.assertRaises(ValueError):
            self.run_migration()
        self.assertFalse(self.target.exists())
        reserved.rmdir()
        descriptor_path = self.source / "w_managed/workspace.json"
        descriptor = json.loads(descriptor_path.read_text())
        descriptor["workspace_id"] = "w_different"
        descriptor_path.write_text(json.dumps(descriptor))
        with self.assertRaises(ValueError):
            self.run_migration()
        self.assertFalse(self.target.exists())

    def test_database_conversion_failure_cleans_staging_and_preserves_source(self):
        before = self.bytes_tree(self.source)
        children = set(self.root.iterdir())
        convert = migration.migrate_database
        writing_calls = []

        def fail_writing(source, target, **kwargs):
            if target is None:
                return convert(source, target, **kwargs)
            writing_calls.append(target)
            raise sqlite3.OperationalError("injected conversion failure")

        with (
            patch.object(
                migration,
                "migrate_database",
                side_effect=fail_writing,
            ),
            self.assertRaises(sqlite3.OperationalError),
        ):
            self.run_migration()
        self.assertTrue(writing_calls)
        self.assertFalse(self.target.exists())
        self.assertEqual(set(self.root.iterdir()), children)
        self.assertEqual(self.bytes_tree(self.source), before)

    def test_target_cannot_overlap_source(self):
        before = self.bytes_tree(self.source)
        for target in (self.source, self.source / "nested", self.source.parent):
            with (
                self.subTest(target=target),
                self.assertRaises((ValueError, FileExistsError)),
            ):
                migration.migrate_workspaces(
                    self.source,
                    target,
                    models=self.models,
                    families=self.families,
                    runners=self.runners,
                    dry_run=False,
                    source_quiesced=True,
                )
        self.assertEqual(self.bytes_tree(self.source), before)

    def test_concurrent_publication_winner_is_never_overwritten_or_deleted(self):
        publish = migration._publish
        before = self.bytes_tree(self.source)

        def competing_target(state, target):
            target.mkdir()
            (target / "winner").write_bytes(b"other migration")
            return publish(state, target)

        with (
            patch.object(migration, "_publish", side_effect=competing_target),
            self.assertRaises(FileExistsError),
        ):
            self.run_migration()
        self.assertEqual(self.bytes_tree(self.target), {"winner": b"other migration"})
        self.assertEqual(self.bytes_tree(self.source), before)

    def test_manifest_inventory_matches_published_tree_including_directory_times(self):
        report = self.run_migration()
        actual = inventory_tree(self.target)
        manifest = next(
            entry
            for entry in actual["entries"]
            if entry["path"] == ".migration-v1/manifest.json"
        )
        actual["entries"] = [
            entry for entry in actual["entries"] if entry is not manifest
        ]
        actual["logical_bytes"] -= manifest["size"]
        actual["unique_bytes"] -= manifest["size"]
        self.assertEqual(report["files"], actual)

    def test_dependency_paths_describe_final_location_and_old_absolute_links(self):
        (self.source / "relative-external").symlink_to(
            "../main", target_is_directory=True
        )
        old_managed = self.source / "w_managed/repo"
        (self.source / "absolute-old").symlink_to(old_managed, target_is_directory=True)
        for dry in (True, False):
            report = self.run_migration(dry_run=dry)
            dependencies = {entry["path"]: entry for entry in report["dependencies"]}
            relative = dependencies["relative-external"]
            self.assertEqual(relative["lexical_target"], str(self.main))
            self.assertTrue(relative["lexically_external"])
            self.assertTrue(relative["requires_validation"])
            absolute = dependencies["absolute-old"]
            self.assertEqual(absolute["lexical_target"], str(old_managed))
            self.assertTrue(absolute["lexically_external"])
            self.assertTrue(absolute["points_to_source"])

    def test_legacy_shared_only_codex_home_is_imported_into_new_role_scope(self):
        shutil.rmtree(self.role)
        report = self.run_migration()
        self.assertTrue(report["verified"])
        home = self.active_home()
        self.assertEqual(
            (home / "sessions/shared.jsonl").read_bytes(), b"shared rollout"
        )
        self.assertEqual(
            (home / "shell_snapshots/environment.sh").read_bytes(), b"shell state"
        )
        self.assertFalse((home / "sessions/local.jsonl").exists())
        self.assertFalse(self.role.exists())

    def test_shared_codex_file_type_conflicts_are_rejected(self):
        collision = self.role / "sessions/shared.jsonl"
        for kind in ("directory", "symlink"):
            if kind == "directory":
                collision.mkdir()
            else:
                collision.symlink_to("nonexistent-other-rollout")
            try:
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    self.run_migration()
                self.assertFalse(self.target.exists())
            finally:
                if collision.is_symlink():
                    collision.unlink()
                else:
                    collision.rmdir()

    def test_registry_disk_identity_set_mismatch_is_rejected(self):
        path = self.source / "registry.json"
        index = json.loads(path.read_text())
        index["workspaces"] = [
            row for row in index["workspaces"] if row["workspace_id"] != "w_managed"
        ]
        path.write_text(json.dumps(index))
        before = self.bytes_tree(self.source)
        with self.assertRaises(ValueError):
            self.run_migration()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.bytes_tree(self.source), before)

    def test_target_inside_old_external_repository_is_rejected(self):
        before = self.bytes_tree(self.main)
        target = self.main / "new-state"
        with self.assertRaises(ValueError):
            migration.migrate_workspaces(
                self.source,
                target,
                models=self.models,
                families=self.families,
                runners=self.runners,
                dry_run=False,
                source_quiesced=True,
            )
        self.assertFalse(target.exists())
        self.assertEqual(self.bytes_tree(self.main), before)

    def test_publish_verification_failure_cleans_readonly_target_but_preserves_source(
        self,
    ):
        readonly = self.source / "readonly"
        readonly.mkdir()
        (readonly / "file").write_bytes(b"read only content")
        readonly.chmod(0o555)
        self.addCleanup(readonly.chmod, 0o755)
        before = self.bytes_tree(self.source)
        inventory = migration.inventory_tree
        checked_target = []

        def fail_final_inventory(path):
            if path == self.target:
                checked_target.append(path)
                raise ValueError("injected final inventory failure")
            return inventory(path)

        with (
            patch.object(migration, "inventory_tree", side_effect=fail_final_inventory),
            self.assertRaisesRegex(ValueError, "injected final inventory failure"),
        ):
            self.run_migration()
        self.assertEqual(checked_target, [self.target])
        self.assertFalse(self.target.exists())
        self.assertEqual(readonly.stat().st_mode & 0o777, 0o555)
        self.assertEqual(self.bytes_tree(self.source), before)
        self.assertTrue(self.run_migration()["verified"])
        self.assertEqual((self.target / "readonly").stat().st_mode & 0o777, 0o555)
        self.assertEqual(
            (self.target / "readonly/file").read_bytes(), b"read only content"
        )

    def test_rehearsal_mismatched_or_symlink_logs_are_rejected(self):
        repo = self.root / "independent-repo"
        logs = repo / "logs"
        logs.mkdir(parents=True)
        separate = self.root / "independent-logs"
        separate.mkdir()
        override = {"w_main": {"repo_path": repo, "logs_path": separate}}
        with self.assertRaises(ValueError):
            self.run_migration(mode="rehearsal", external_paths=override)
        self.assertFalse(self.target.exists())
        logs.rmdir()
        logs.symlink_to(separate, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.run_migration(mode="rehearsal", external_paths=override)
        self.assertFalse(self.target.exists())
        self.assertTrue(logs.is_symlink())

    def test_readonly_shared_codex_directories_merge_into_existing_role_sessions(self):
        shared = self.legacy / "sessions"
        nested = shared / "nested"
        nested.mkdir()
        (nested / "rollout.jsonl").write_bytes(b"nested shared rollout")
        role_nested = self.role / "sessions/nested"
        role_nested.mkdir()
        (role_nested / "local.jsonl").write_bytes(b"local nested rollout")
        role_nested.chmod(0o555)
        shared.chmod(0o555)
        nested.chmod(0o555)
        self.addCleanup(shared.chmod, 0o755)
        self.addCleanup(nested.chmod, 0o755)
        self.addCleanup(role_nested.chmod, 0o755)
        before = self.bytes_tree(self.source)
        self.run_migration()
        self.assertEqual(
            (self.active_home() / "sessions/local.jsonl").read_bytes(), b"local rollout"
        )
        self.assertEqual(
            (self.active_home() / "sessions/shared.jsonl").read_bytes(),
            b"shared rollout",
        )
        self.assertEqual(
            (self.active_home() / "sessions/nested/rollout.jsonl").read_bytes(),
            b"nested shared rollout",
        )
        self.assertEqual(
            (self.active_home() / "sessions/nested").stat().st_mode & 0o777, 0o555
        )
        self.assertEqual(
            (self.active_home() / "sessions/nested/local.jsonl").read_bytes(),
            b"local nested rollout",
        )
        self.assertEqual(shared.stat().st_mode & 0o777, 0o555)
        self.assertEqual(nested.stat().st_mode & 0o777, 0o555)
        self.assertEqual(self.bytes_tree(self.source), before)

    def test_unrecognized_binary_git_file_is_preserved_and_reported_unverified(self):
        artifact = self.source / "unknown-artifact"
        artifact.mkdir()
        content = b"not a git pointer\x00\xff"
        (artifact / ".git").write_bytes(content)
        report = self.run_migration()
        self.assertEqual((self.target / "unknown-artifact/.git").read_bytes(), content)
        entry = next(
            item
            for item in report["dependencies"]
            if item["path"] == "unknown-artifact/.git"
        )
        self.assertTrue(entry["unrecognized"])
        self.assertTrue(entry["requires_validation"])
        self.assertFalse(report["execution_isolation_verified"])

    def test_malformed_runner_cli_uses_fixed_error_without_disclosing_value(self):
        path = self.root / "mapping.json"
        path.write_text(
            json.dumps(
                {
                    "models": {},
                    "families": {},
                    "runners": {"old-family": ["private-runner-value"]},
                }
            )
        )
        stderr, stdout = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(stdout),
            self.assertRaises(SystemExit) as caught,
        ):
            migration.main(
                [
                    str(self.source),
                    str(self.target),
                    "--mapping",
                    str(path),
                    "--apply",
                    "--source-quiesced",
                ]
            )
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(
            "cannot read a valid workspace migration mapping", stderr.getvalue()
        )
        self.assertNotIn("private-runner-value", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertFalse(self.target.exists())
