"""Database creation and transactions, separate from offline schema upgrades."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

APPLICATION_ID = 0x56424132
SCHEMA_VERSION = 1


class SchemaMismatch(ValueError):
    """The database needs an explicit offline conversion before it can be used."""


class Database:
    def __init__(self, path: Path):
        self.path = path.resolve()

    @classmethod
    def create(cls, path: Path) -> Database:
        database = cls(path)
        database.path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only the initialized database, without replacing another creator.
        with TemporaryDirectory(prefix=".agent-db-", dir=database.path.parent) as staging:
            staged_path = Path(staging) / database.path.name
            with closing(sqlite3.connect(staged_path)) as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA foreign_keys = ON")
                schema = Path(__file__).with_name("schema.sql").read_text()
                connection.executescript(
                    "BEGIN IMMEDIATE;\n" + schema
                    + f"\nPRAGMA application_id = {APPLICATION_ID};"
                    + f"\nPRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;"
                )
            # Closing the sole connection checkpoints WAL before moving the inode.
            os.link(staged_path, database.path)
        return database

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        mode = "rw" if write else "ro"
        with closing(sqlite3.connect(self.path.as_uri() + f"?mode={mode}", uri=True, timeout=30)) as connection:
            connection.row_factory = sqlite3.Row
            application = connection.execute("PRAGMA application_id").fetchone()[0]
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if application != APPLICATION_ID or version != SCHEMA_VERSION:
                raise SchemaMismatch(
                    f"Unsupported Agent database format ({application}, {version}); "
                    "run the offline migration before starting the service"
                )
            connection.execute("PRAGMA foreign_keys = ON")
            if not write:
                connection.execute("PRAGMA query_only = ON")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
