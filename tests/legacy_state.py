"""Build synthetic legacy state from the frozen, independently generated v8 DDL."""

import sqlite3
from contextlib import closing
from pathlib import Path


def create_legacy_database(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=False)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            (Path(__file__).parent / "fixtures/legacy_v8/schema.sql").read_text()
        )
        connection.executemany(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            [(version, float(version)) for version in (1, 8)],
        )
    return path
