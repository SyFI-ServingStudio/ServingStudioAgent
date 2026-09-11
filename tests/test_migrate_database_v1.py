import contextlib
import hashlib
import io
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.legacy_workspaces import create_legacy_workspace
from tools import migrate_v1_database as migration
from vibesim_agent.storage.database import Database


class DatabaseMigrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        main = self.root / "main"
        main.mkdir()
        self.source = create_legacy_workspace(self.root / "legacy", main)
        self.target = self.root / "converted.sqlite"
        self.writer = sqlite3.connect(self.source)
        self.addCleanup(self.writer.close)
        self.writer.execute("PRAGMA wal_autocheckpoint=0")
        self.writer.execute("PRAGMA foreign_keys=ON")
        self.writer.execute("""INSERT INTO conversations
            (id,title,naming_state,sandbox,autonomous,agent_mode,interrupted_role,
             orchestrator_model,orchestrator_effort,orchestrator_service_tier,
             implementer_model,implementer_effort,implementer_service_tier,
             assistant_model,assistant_effort,assistant_service_tier,
             prompt_fingerprint,peer_workspace,created_at,updated_at)
             VALUES ('c','Original','manual','workspace-write',1,'single','assistant',
                     'old-model','raw-effort','unusual-tier',
                     'old-model','raw-effort','unusual-tier',
                     'old-model','raw-effort','unusual-tier',NULL,NULL,1.25,9.5)""")
        self.writer.execute("INSERT INTO turns VALUES ('t','c','running',2.5,3.75)")
        self.writer.execute(
            "INSERT INTO codex_sessions VALUES ('c','assistant','old-family','resume-id')"
        )
        self.writer.execute(
            "INSERT INTO messages VALUES (7,'c','user',?,4.25,?,NULL)",
            (sqlite3.Binary(b"raw\x00\xff"), ' {"unicode":"\\u4f60", "n":1.0} '),
        )
        self.writer.execute(
            "INSERT INTO messages VALUES (19,'c','assistant','saved',5.5,'{}','t')"
        )
        self.writer.execute(
            "INSERT INTO turn_events VALUES (12,'t',3,'progress',?)",
            (' {"text": "one"} ',),
        )
        self.writer.execute(
            "INSERT INTO turn_events VALUES (44,'t',17,'opaque',?)",
            (sqlite3.Binary(b"opaque\xff"),),
        )
        self.writer.execute(
            "INSERT INTO experiments VALUES ('e','result','ready','agent','deleted-job',1.0,7.0)"
        )
        self.writer.execute("""INSERT INTO execution_jobs
            (id,conversation_id,turn_id,role,status,job_kind,artifact_path,resource_id,
             analyzer_resource_id,descriptor_json,summary_json,experiment_id,experiment_path,created_at,updated_at)
            VALUES ('j','c','t','assistant','requested','simulation',NULL,NULL,NULL,' {} ',NULL,'e','result',3,4)""")
        self.writer.execute(
            "INSERT INTO conversation_experiments VALUES ('c','e',NULL,'viewed')"
        )
        self.writer.execute("UPDATE sqlite_sequence SET seq=900 WHERE name='messages'")
        self.writer.execute(
            "UPDATE sqlite_sequence SET seq=700 WHERE name='turn_events'"
        )
        self.writer.execute("INSERT OR REPLACE INTO schema_migrations VALUES (3,1.125)")
        self.writer.commit()
        self.models = {
            "old-model": migration.ProviderIdentity("model_provider", "model:scope")
        }
        self.families = {
            "old-family": migration.ProviderIdentity(
                "session_provider", "session:scope"
            )
        }

    def migrate(self, target=True, **kwargs):
        return migration.migrate_database(
            self.source,
            self.target if target else None,
            models=kwargs.get("models", self.models),
            families=kwargs.get("families", self.families),
        )

    def source_bytes(self):
        return self.source.read_bytes(), Path(str(self.source) + "-wal").read_bytes()

    def rows(self, connection, table):
        order = "name" if table == "sqlite_sequence" else "rowid"
        return [
            tuple(row)
            for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY {order}')
        ]

    def test_wal_snapshot_preserves_raw_tables_sparse_ids_highwater_and_independent_mappings(
        self,
    ):
        before = self.source_bytes()
        self.migrate()
        with Database(self.target).connect() as converted:
            for table in (
                "messages",
                "turns",
                "turn_events",
                "execution_jobs",
                "experiments",
                "conversation_experiments",
                "sqlite_sequence",
            ):
                self.assertEqual(
                    self.rows(converted, table), self.rows(self.writer, table), table
                )
            runtimes = converted.execute(
                "SELECT role,provider_id,session_scope,model_id,effort,service_tier FROM role_settings ORDER BY role"
            ).fetchall()
            self.assertEqual(
                [tuple(row) for row in runtimes],
                [
                    (
                        role,
                        "model_provider",
                        "model:scope",
                        "old-model",
                        "raw-effort",
                        "unusual-tier",
                    )
                    for role in ("assistant", "implementer", "orchestrator")
                ],
            )
            self.assertEqual(
                tuple(converted.execute("SELECT * FROM agent_sessions").fetchone()),
                ("c", "assistant", "session_provider", "session:scope", "resume-id"),
            )
            columns = [
                row[1] for row in converted.execute("PRAGMA table_info(conversations)")
            ]
            selected = ",".join(columns)
            self.assertEqual(
                tuple(
                    converted.execute(
                        f"SELECT {selected} FROM conversations"
                    ).fetchone()
                ),
                tuple(
                    self.writer.execute(
                        f"SELECT {selected} FROM conversations"
                    ).fetchone()
                ),
            )
        self.assertEqual(self.source_bytes(), before)

    def test_dry_run_checks_conversion_without_target_or_source_changes(self):
        before = self.source_bytes()
        paths = set(self.root.rglob("*"))
        report = self.migrate(target=False)
        self.assertIsInstance(report, dict)
        self.assertEqual(set(self.root.rglob("*")), paths)
        self.assertEqual(self.source_bytes(), before)
        self.assertNotIn("resume-id", repr(report))

    def test_empty_autoincrement_tables_preserve_sequence_presence_and_values(self):
        self.writer.execute("DELETE FROM messages")
        self.writer.execute("DELETE FROM turn_events")
        self.writer.commit()
        self.migrate()
        with Database(self.target).connect(write=True) as converted:
            self.assertEqual(
                self.rows(converted, "sqlite_sequence"),
                self.rows(self.writer, "sqlite_sequence"),
            )
            message = converted.execute(
                "INSERT INTO messages(conversation_id,role,content,ts,metadata_json) VALUES ('c','user','next',10,'{}')"
            )
            event = converted.execute(
                "INSERT INTO turn_events(turn_id,sequence,kind,payload_json) VALUES ('t',0,'next','{}')"
            )
            self.assertEqual(message.lastrowid, 901)
            self.assertEqual(event.lastrowid, 701)

    def test_unknown_model_or_family_is_rejected_even_in_dry_run(self):
        for mappings in ({"models": {}}, {"families": {}}):
            before = self.source_bytes()
            with self.subTest(mappings=mappings), self.assertRaises(ValueError):
                self.migrate(target=False, **mappings)
            self.assertEqual(self.source_bytes(), before)
            self.assertFalse(self.target.exists())

    def test_existing_target_symlink_and_source_identity_are_never_overwritten(self):
        self.target.write_bytes(b"existing")
        with self.assertRaises((ValueError, FileExistsError)):
            self.migrate()
        self.assertEqual(self.target.read_bytes(), b"existing")
        self.target.unlink()
        elsewhere = self.root / "nonexistent"
        self.target.symlink_to(elsewhere)
        with self.assertRaises((ValueError, FileExistsError)):
            self.migrate()
        self.assertTrue(self.target.is_symlink())
        self.assertFalse(elsewhere.exists())
        before = self.source_bytes()
        with self.assertRaises((ValueError, FileExistsError)):
            migration.migrate_database(
                self.source, self.source, models=self.models, families=self.families
            )
        self.assertEqual(self.source_bytes(), before)

    def test_unknown_schema_objects_and_columns_are_not_silently_dropped(self):
        for statement in (
            "CREATE TABLE future(value TEXT)",
            "ALTER TABLE conversations ADD COLUMN future TEXT",
            "CREATE VIEW future AS SELECT id FROM conversations",
            "CREATE TRIGGER future AFTER UPDATE ON conversations BEGIN SELECT 1; END",
        ):
            self.writer.execute(statement)
            self.writer.commit()
            before = self.source_bytes()
            with self.subTest(statement=statement), self.assertRaises(ValueError):
                self.migrate()
            self.assertFalse(self.target.exists())
            self.assertEqual(self.source_bytes(), before)
            if statement.startswith("ALTER"):
                self.writer.execute("ALTER TABLE conversations DROP COLUMN future")
            else:
                object_type = statement.split()[1]
                self.writer.execute(f"DROP {object_type} future")
            self.writer.commit()

    def test_real_foreign_key_and_cross_conversation_turn_defects_are_rejected(self):
        columns = [
            row[1]
            for row in self.writer.execute("PRAGMA table_info(conversations)")
            if row[1] != "id"
        ]
        copied = ",".join(columns)
        self.writer.execute(
            f"INSERT INTO conversations(id,{copied}) SELECT 'other',{copied} FROM conversations WHERE id='c'"
        )
        self.writer.execute(
            "INSERT INTO turns VALUES ('foreign-turn','other','complete',1,2)"
        )
        self.writer.commit()
        self.writer.execute("PRAGMA foreign_keys=OFF")
        for statement, restore in (
            (
                "UPDATE messages SET conversation_id='missing' WHERE id=7",
                "UPDATE messages SET conversation_id='c' WHERE id=7",
            ),
            (
                "UPDATE execution_jobs SET turn_id='missing'",
                "UPDATE execution_jobs SET turn_id='t'",
            ),
            (
                "UPDATE messages SET turn_id='missing' WHERE id=19",
                "UPDATE messages SET turn_id='t' WHERE id=19",
            ),
            (
                "UPDATE execution_jobs SET turn_id='foreign-turn'",
                "UPDATE execution_jobs SET turn_id='t'",
            ),
            (
                "UPDATE messages SET turn_id='foreign-turn' WHERE id=19",
                "UPDATE messages SET turn_id='t' WHERE id=19",
            ),
            (
                "UPDATE conversation_experiments SET turn_id='foreign-turn'",
                "UPDATE conversation_experiments SET turn_id=NULL",
            ),
        ):
            self.writer.execute(statement)
            self.writer.commit()
            with self.subTest(statement=statement), self.assertRaises(ValueError):
                self.migrate()
            self.assertFalse(self.target.exists())
            self.writer.execute(restore)
            self.writer.commit()

    def test_target_schema_failure_leaves_no_published_file_or_changed_source(self):
        before = self.source_bytes()
        with (
            patch.object(
                Database,
                "create",
                side_effect=sqlite3.OperationalError("injected SQL failure"),
            ),
            self.assertRaises(sqlite3.OperationalError),
        ):
            self.migrate()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.source_bytes(), before)

    def test_absent_sequence_entry_stays_absent_for_empty_table(self):
        self.writer.execute("DELETE FROM turn_events")
        self.writer.execute("DELETE FROM sqlite_sequence WHERE name='turn_events'")
        self.writer.commit()
        self.migrate()
        with Database(self.target).connect() as converted:
            self.assertEqual(
                self.rows(converted, "sqlite_sequence"),
                self.rows(self.writer, "sqlite_sequence"),
            )

    def test_unmigrated_schema_version_is_rejected_without_target(self):
        self.writer.execute("DELETE FROM schema_migrations WHERE version>=8")
        self.writer.commit()
        before = self.source_bytes()
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.source_bytes(), before)

    def test_report_preserves_versions_mapping_and_typed_digests_without_scope_values(
        self,
    ):
        report = self.migrate(target=False)
        self.assertFalse(report["verified"])
        self.assertEqual(report["snapshot"], "sqlite-read-transaction")
        self.assertEqual(
            report["schema_migrations"],
            self.writer.execute(
                "SELECT version,applied_at FROM schema_migrations ORDER BY version"
            ).fetchall(),
        )
        self.assertEqual(
            report["relationship_warnings"], {"experiments.job_id.orphaned": 1}
        )
        self.assertEqual(
            report["mapping"]["models"]["old-model"],
            {
                "provider_id": "model_provider",
                "session_scope_sha256": hashlib.sha256(b"model:scope").hexdigest(),
            },
        )
        self.assertNotIn("session:scope", repr(report))
        self.assertEqual(report["tables"]["messages"]["count"], 2)
        converted = self.migrate()
        self.assertTrue(converted["verified"])
        self.assertEqual(converted["tables"], report["tables"])
        original = self.writer.execute(
            "SELECT metadata_json FROM messages WHERE id=7"
        ).fetchone()[0]
        self.writer.execute(
            "UPDATE messages SET metadata_json=? WHERE id=7",
            (sqlite3.Binary(original.encode()),),
        )
        self.writer.commit()
        changed = self.migrate(target=False)
        self.assertNotEqual(
            changed["tables"]["messages"]["sha256"],
            report["tables"]["messages"]["sha256"],
        )
        self.assertEqual(
            changed["tables"]["turn_events"], report["tables"]["turn_events"]
        )

    def test_empty_legacy_session_family_uses_explicit_independent_mapping(self):
        self.writer.execute("UPDATE codex_sessions SET family=''")
        self.writer.commit()
        self.migrate(
            families={
                "": migration.ProviderIdentity("session_provider", "legacy:empty")
            }
        )
        with Database(self.target).connect() as converted:
            self.assertEqual(
                tuple(
                    converted.execute(
                        "SELECT provider_id,session_scope FROM agent_sessions"
                    ).fetchone()
                ),
                ("session_provider", "legacy:empty"),
            )
            self.assertEqual(
                tuple(
                    converted.execute(
                        "SELECT provider_id,session_scope FROM role_settings LIMIT 1"
                    ).fetchone()
                ),
                ("model_provider", "model:scope"),
            )

    def test_unknown_generated_column_cannot_be_silently_omitted(self):
        self.writer.execute(
            "ALTER TABLE messages ADD COLUMN future TEXT GENERATED ALWAYS AS (content) VIRTUAL"
        )
        self.writer.commit()
        before = self.source_bytes()
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.source_bytes(), before)

    def test_copy_failure_and_corrupted_target_validation_do_not_publish(self):
        create = Database.create
        for body, error in (
            ("SELECT RAISE(ABORT, 'injected copy failure');", sqlite3.IntegrityError),
            ("UPDATE messages SET content='corrupted' WHERE id=NEW.id;", ValueError),
        ):

            def staged(path, body=body):
                database = create(path)
                with database.connect(write=True) as connection:
                    connection.execute(
                        "CREATE TRIGGER inject AFTER INSERT ON messages BEGIN "
                        + body
                        + " END"
                    )
                return database

            before = self.source_bytes()
            paths = set(self.root.iterdir())
            with (
                self.subTest(body=body),
                patch.object(Database, "create", side_effect=staged),
                self.assertRaises(error),
            ):
                self.migrate()
            self.assertFalse(self.target.exists())
            self.assertEqual(set(self.root.iterdir()), paths)
            self.assertEqual(self.source_bytes(), before)

    def test_source_writer_commit_between_digest_and_copy_cannot_change_snapshot(self):
        create = Database.create
        old_messages = self.rows(self.writer, "messages")
        old_events = self.rows(self.writer, "turn_events")
        old_sequences = self.rows(self.writer, "sqlite_sequence")

        def concurrent_write(path):
            staged = create(path)
            self.writer.execute(
                "INSERT INTO messages(conversation_id,role,content,ts,metadata_json) VALUES ('c','user','late',99,'{}')"
            )
            self.writer.execute(
                "INSERT INTO turn_events(turn_id,sequence,kind,payload_json) VALUES ('t',99,'late','{}')"
            )
            self.writer.commit()
            return staged

        with patch.object(Database, "create", side_effect=concurrent_write):
            report = self.migrate()
        self.assertTrue(report["verified"])
        self.assertEqual(len(self.rows(self.writer, "messages")), len(old_messages) + 1)
        with Database(self.target).connect() as converted:
            self.assertEqual(self.rows(converted, "messages"), old_messages)
            self.assertEqual(self.rows(converted, "turn_events"), old_events)
            self.assertEqual(self.rows(converted, "sqlite_sequence"), old_sequences)

    def test_unserializable_migration_metadata_is_rejected_before_publication(self):
        for value in (sqlite3.Binary(b"private-metadata"), float("inf")):
            self.writer.execute(
                "UPDATE schema_migrations SET applied_at=? WHERE version=8", (value,)
            )
            self.writer.commit()
            before = self.source_bytes()
            with (
                self.subTest(value_type=type(value).__name__),
                self.assertRaises(ValueError),
            ):
                self.migrate()
            self.assertFalse(self.target.exists())
            self.assertEqual(self.source_bytes(), before)

    def test_malformed_mapping_cli_error_never_echoes_private_values(self):
        document = self.root / "mapping.json"
        for content in (
            '{"private-secret":',
            '{"models":{"m":{"provider_id":"PRIVATE-SECRET","session_scope":"private-scope"}},"families":{}}',
        ):
            document.write_text(content)
            stderr, stdout = io.StringIO(), io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                contextlib.redirect_stdout(stdout),
                self.assertRaises(SystemExit) as caught,
            ):
                migration.main(
                    [
                        str(self.source),
                        "--target",
                        str(self.target),
                        "--mapping",
                        str(document),
                    ]
                )
            self.assertEqual(caught.exception.code, 2)
            self.assertIn(
                "cannot read a valid provider mapping document", stderr.getvalue()
            )
            self.assertNotIn("PRIVATE-SECRET", stderr.getvalue())
            self.assertNotIn("private-secret", stderr.getvalue())
            self.assertNotIn("private-scope", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse(self.target.exists())
