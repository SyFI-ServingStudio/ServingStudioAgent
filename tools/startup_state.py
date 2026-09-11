"""Classify all workspace databases before a managed startup chooses migration.

This read-only preflight does not establish quiescence. Database files and WALs
are inspected in scratch copies so opening SQLite cannot update source SHM.
"""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

from tools.migrate_v1_database import (
    MigrationError,
    _columns,
    _foreign_keys,
    _shape,
    _uniques,
    _validate_source,
)
from tools.migrate_v1_workspaces import _database_scratch, _descriptors
from tools.snapshot_v1 import relationships
from vibesim_agent.storage.database import APPLICATION_ID, SCHEMA_VERSION


def _validate_current(connection, expected):
    objects = connection.execute(
        "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'view', 'trigger')"
    ).fetchall()
    reference = expected.execute(
        "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'view', 'trigger')"
    ).fetchall()
    if set(objects) != set(reference):
        raise MigrationError("unsupported current database objects")
    for _, table in reference:
        columns = _columns(connection, table)
        if (
            any(column[6] for column in columns)
            or _shape(columns) != _shape(_columns(expected, table))
            or _uniques(connection, table) != _uniques(expected, table)
            or _foreign_keys(connection, table) != _foreign_keys(expected, table)
        ):
            raise MigrationError("unsupported current database columns or constraints")
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise MigrationError("current database failed SQLite integrity_check")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise MigrationError("current database has foreign key violations")
    links = relationships(
        connection,
        {table: {"columns": _columns(connection, table)} for _, table in reference},
    )
    invalid = {
        name
        for name, count in links.items()
        if count and name != "experiments.job_id.orphaned"
    }
    if invalid:
        raise MigrationError(
            "current database has invalid relationships: " + ", ".join(sorted(invalid))
        )


def inspect_state(root: Path) -> Literal["current", "legacy-v8"]:
    """Reject missing, mixed, damaged or unknown state without migrating it."""
    if root.is_symlink() or not root.is_dir():
        raise MigrationError("startup requires an existing real workspace state root")
    root = root.resolve()
    try:
        descriptors = _descriptors(root)
        formats = set()
        with closing(sqlite3.connect(":memory:")) as expected:
            expected.executescript(
                Path(__file__)
                .parents[1]
                .joinpath("vibesim_agent/storage/schema.sql")
                .read_text()
            )
            for identity in descriptors:
                with (
                    _database_scratch(root / identity) as scratch,
                    closing(
                        sqlite3.connect(scratch.as_uri() + "?mode=ro", uri=True)
                    ) as connection,
                ):
                    connection.execute("PRAGMA query_only = ON")
                    connection.execute("BEGIN")
                    version = (
                        connection.execute("PRAGMA application_id").fetchone()[0],
                        connection.execute("PRAGMA user_version").fetchone()[0],
                    )
                    if version == (APPLICATION_ID, SCHEMA_VERSION):
                        _validate_current(connection, expected)
                        formats.add("current")
                    elif version == (0, 0):
                        _validate_source(connection, expected)
                        formats.add("legacy-v8")
                    else:
                        raise MigrationError("unsupported workspace database version")
        if len(formats) != 1:
            raise MigrationError(
                "mixed workspace database formats require repair before startup"
            )
        return formats.pop()
    except sqlite3.DatabaseError as error:
        raise MigrationError("workspace database could not be validated") from error
