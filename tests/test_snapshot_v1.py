"""The migration inventory must observe committed WAL rows without writing."""

import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.snapshot_v1 import database, routes


class InventoryTests(unittest.TestCase):
    def test_cli_rejects_missing_or_nonlegacy_routes_without_writing_baseline(self):
        script = Path(__file__).resolve().parents[1] / "tools/snapshot_v1.py"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output/baseline.json"
            for source, message in (
                (None, "retained legacy Agent checkout"),
                (
                    'raise RuntimeError("must not execute")\n',
                    "no supported @app routes",
                ),
            ):
                with self.subTest(source=source):
                    if source is not None:
                        (root / "backend").mkdir()
                        (root / "backend/app.py").write_text(source)
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(script),
                            "--repo",
                            str(root),
                            "--output",
                            str(output),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(output.parent.exists())

    def test_undeclared_turn_links_are_audited_without_inventing_legacy_links(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "workspace.sqlite"
            with sqlite3.connect(path) as writer:
                writer.executescript("""
                    CREATE TABLE turns (id TEXT, conversation_id TEXT);
                    CREATE TABLE messages (id INTEGER, conversation_id TEXT, turn_id TEXT);
                    INSERT INTO turns VALUES ('turn', 'owner');
                    INSERT INTO messages VALUES (1, 'owner', NULL);
                    INSERT INTO messages VALUES (2, 'owner', 'missing');
                    INSERT INTO messages VALUES (3, 'other', 'turn');
                    INSERT INTO messages VALUES (4, 'owner', 'turn');
                """)
            self.assertEqual(
                database(path)["relationship_errors"],
                {
                    "messages.turn_id.orphaned": 1,
                    "messages.turn_id.conversation_mismatch": 1,
                },
            )

    def test_wal_inventory_is_read_only_and_includes_ids_and_sequence(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "workspace.sqlite"
            with sqlite3.connect(path) as writer:
                writer.execute("PRAGMA journal_mode=WAL")
                writer.execute(
                    "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)"
                )
                writer.execute(
                    "INSERT INTO messages (id, content) VALUES (42, 'private message')"
                )
                writer.commit()
                before = path.read_bytes(), Path(str(path) + "-wal").read_bytes()
                captured = database(path)
                self.assertEqual(captured["tables"]["messages"]["count"], 1)
                self.assertEqual(captured["tables"]["sqlite_sequence"]["count"], 1)
                self.assertNotIn("private message", str(captured))
                self.assertEqual(captured, database(path))
                self.assertEqual(
                    before, (path.read_bytes(), Path(str(path) + "-wal").read_bytes())
                )
                writer.execute("UPDATE messages SET content = 'changed' WHERE id = 42")
                writer.commit()
                self.assertNotEqual(
                    database(path)["tables"]["messages"]["sha256"],
                    captured["tables"]["messages"]["sha256"],
                )

    def test_route_inventory_does_not_execute_source(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "app.py"
            source.write_text(
                'raise RuntimeError("must not import")\n'
                '@app.post("/tools/example")\n'
                "def example(body, auth=Depends(require_token)): pass\n"
            )
            self.assertEqual(
                routes(source),
                [
                    {
                        "method": "POST",
                        "path": "/tools/example",
                        "handler": "example",
                        "dependencies": ["require_token"],
                    }
                ],
            )
