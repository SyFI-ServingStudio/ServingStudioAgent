"""Offline conversion of one legacy v8 database from a single read snapshot.

This does not freeze workspace files or authorize provider session compatibility.
The caller must persist the returned manifest alongside the workspace migration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from tools.snapshot_v1 import relationships
from vibesim_agent.domain.roles import Role
from vibesim_agent.settings import validate_provider_id
from vibesim_agent.storage.database import Database


class MigrationError(ValueError):
    """The source or explicit compatibility mapping cannot be converted losslessly."""


@dataclass(frozen=True)
class ProviderIdentity:
    provider_id: str
    session_scope: str = field(repr=False)

    def __post_init__(self):
        validate_provider_id(self.provider_id)
        if not isinstance(self.session_scope, str) or not self.session_scope:
            raise MigrationError("provider mapping requires a nonempty session scope")


ORDER = {
    "conversations": ("id",),
    "role_settings": ("conversation_id", "role"),
    "agent_sessions": ("conversation_id", "role"),
    "messages": ("id",),
    "turns": ("id",),
    "turn_events": ("id",),
    "execution_jobs": ("id",),
    "experiments": ("id",),
    "conversation_experiments": ("conversation_id", "experiment_id", "relation"),
    "sqlite_sequence": ("name", "rowid"),
}
ROLE_FIELDS = ("model", "effort", "service_tier")


def _columns(connection, table):
    return connection.execute(f'PRAGMA table_xinfo("{table}")').fetchall()


def _uniques(connection, table):
    result = set()
    for index in connection.execute(f'PRAGMA index_list("{table}")'):
        if index[2]:
            name = index[1].replace('"', '""')
            columns = tuple(
                (row[2], row[3], row[4])
                for row in connection.execute(f'PRAGMA index_xinfo("{name}")')
                if row[5]
            )
            result.add((columns, index[4]))
    return result


def _foreign_keys(connection, table):
    return sorted(
        tuple(row)[2:]
        for row in connection.execute(f'PRAGMA foreign_key_list("{table}")')
    )


def _shape(columns):
    return {row[1]: (row[2].upper(), row[3], row[5]) for row in columns}


def _validate_source(source, target_schema):
    schema = source.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()
    expected = set(ORDER) - {"role_settings", "agent_sessions"} | {
        "codex_sessions",
        "schema_migrations",
    }
    if {row[1] for row in schema if row[0] == "table"} != expected:
        raise MigrationError("unsupported legacy table set")
    if any(row[0] in {"trigger", "view"} for row in schema):
        raise MigrationError("legacy triggers or views require explicit conversion")
    if (
        source.execute("PRAGMA application_id").fetchone()[0]
        or source.execute("PRAGMA user_version").fetchone()[0]
    ):
        raise MigrationError("source is not the supported legacy database format")
    info = {table: _columns(source, table) for table in expected}
    if any(column[6] for columns in info.values() for column in columns):
        raise MigrationError(
            "generated or hidden legacy columns require explicit conversion"
        )
    for table in expected:
        target = "agent_sessions" if table == "codex_sessions" else table
        if table == "schema_migrations":
            shape = {"version": ("INTEGER", 0, 1), "applied_at": ("REAL", 1, 0)}
        else:
            shape = _shape(_columns(target_schema, target))
            if table == "conversations":
                shape.update(
                    {
                        f"{role.value}_{name}": ("TEXT", 1, 0)
                        for role in Role
                        for name in ROLE_FIELDS
                    }
                )
            elif table == "codex_sessions":
                del shape["provider_id"], shape["session_scope"]
                shape["family"] = ("TEXT", 1, 0)
        if _shape(info[table]) != shape:
            raise MigrationError(f"unsupported legacy columns or constraints: {table}")
        if table != "schema_migrations" and (
            _uniques(source, table) != _uniques(target_schema, target)
            or _foreign_keys(source, table) != _foreign_keys(target_schema, target)
        ):
            raise MigrationError(f"unsupported legacy relational constraints: {table}")
    migrations = [
        tuple(row)
        for row in source.execute(
            "SELECT version, applied_at FROM schema_migrations ORDER BY version"
        )
    ]
    if (
        not migrations
        or migrations[-1][0] != 8
        or any(type(row[0]) is not int or not 1 <= row[0] <= 8 for row in migrations)
    ):
        raise MigrationError("source requires the supported legacy v8 schema")
    if [row[0] for row in source.execute("PRAGMA integrity_check")] != ["ok"]:
        raise MigrationError("source failed SQLite integrity_check")
    if source.execute("PRAGMA foreign_key_check").fetchall():
        raise MigrationError("source has foreign key violations")
    links = relationships(
        source, {name: {"columns": columns} for name, columns in info.items()}
    )
    invalid = {
        name: count
        for name, count in links.items()
        if count and name != "experiments.job_id.orphaned"
    }
    if invalid:
        raise MigrationError(
            "source has invalid relationships: " + ", ".join(sorted(invalid))
        )
    return schema, migrations, {name: count for name, count in links.items() if count}


def _rows(connection, table, columns):
    names = ", ".join(f'"{name}"' for name in columns)
    order = ", ".join(f'"{name}"' for name in ORDER[table])
    return connection.execute(f'SELECT {names} FROM "{table}" ORDER BY {order}')


def _mapped_rows(source, table, columns, models, families) -> Iterator[tuple]:
    if table == "role_settings":
        for row in source.execute("SELECT * FROM conversations ORDER BY id"):
            for role in sorted(Role, key=lambda role: role.value):
                model = row[f"{role.value}_model"]
                identity = models.get(model)
                if identity is None:
                    raise MigrationError("missing explicit model mapping")
                yield (
                    row["id"],
                    role.value,
                    identity.provider_id,
                    identity.session_scope,
                    model,
                    row[f"{role.value}_effort"],
                    row[f"{role.value}_service_tier"],
                )
    elif table == "agent_sessions":
        for row in source.execute(
            "SELECT conversation_id, role, family, session_id FROM codex_sessions ORDER BY conversation_id, role"
        ):
            identity = families.get(row["family"])
            if identity is None:
                raise MigrationError("missing explicit session family mapping")
            if row["role"] not in {role.value for role in Role}:
                raise MigrationError("unsupported legacy session role")
            yield (
                row["conversation_id"],
                row["role"],
                identity.provider_id,
                identity.session_scope,
                row["session_id"],
            )
    else:
        yield from (tuple(row) for row in _rows(source, table, columns))


def _digest(rows) -> dict[str, Any]:
    result = hashlib.sha256()
    count = 0
    for row in rows:
        encoded = []
        for value in row:
            if value is None:
                item = ["null"]
            elif isinstance(value, bytes):
                item = ["blob", value.hex()]
            elif isinstance(value, str):
                item = ["text", value]
            elif isinstance(value, int):
                item = ["integer", str(value)]
            elif isinstance(value, float):
                item = ["real", value.hex()]
            else:
                raise MigrationError("unsupported SQLite value type")
            encoded.append(item)
        result.update(
            json.dumps(encoded, ensure_ascii=True, separators=(",", ":")).encode()
        )
        result.update(b"\n")
        count += 1
    return {"count": count, "sha256": result.hexdigest()}


def _mapping_manifest(mapping, *, allow_empty: bool = False):
    result = {}
    for key, identity in sorted(mapping.items()):
        if (
            not isinstance(key, str)
            or (not key and not allow_empty)
            or not isinstance(identity, ProviderIdentity)
        ):
            raise MigrationError("invalid explicit provider mapping")
        result[key] = {
            "provider_id": identity.provider_id,
            "session_scope_sha256": hashlib.sha256(
                identity.session_scope.encode()
            ).hexdigest(),
        }
    return result


def migrate_database(
    source: Path,
    target: Path | None,
    *,
    models: Mapping[str, ProviderIdentity],
    families: Mapping[str, ProviderIdentity],
) -> dict:
    source = source.resolve(strict=True)
    if target is not None:
        if target.exists() or target.is_symlink():
            raise FileExistsError("migration target already exists")
        target = target.resolve()
        if target == source or target in {
            Path(str(source) + suffix) for suffix in ("-wal", "-shm", "-journal")
        }:
            raise MigrationError("migration target overlaps source database state")
    mapping = {
        "models": _mapping_manifest(models),
        "families": _mapping_manifest(families, allow_empty=True),
    }
    with (
        closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original,
        closing(sqlite3.connect(":memory:")) as schema,
    ):
        original.row_factory = sqlite3.Row
        original.execute("PRAGMA query_only = ON")
        original.execute("BEGIN")
        schema.executescript(
            Path(__file__)
            .parents[1]
            .joinpath("vibesim_agent/storage/schema.sql")
            .read_text()
        )
        old_schema, migrations, warnings = _validate_source(original, schema)
        columns = {
            table: tuple(row[1] for row in _columns(schema, table)) for table in ORDER
        }
        expected = {
            table: _digest(
                _mapped_rows(original, table, columns[table], models, families)
            )
            for table in ORDER
        }
        report = {
            "source": str(source),
            "target": None if target is None else str(target),
            "snapshot": "sqlite-read-transaction",
            "schema_migrations": migrations,
            "source_schema": _digest(old_schema),
            "mapping": mapping,
            "tables": expected,
            "relationship_warnings": warnings,
            "verified": False,
        }
        try:
            json.dumps(report, allow_nan=False)
        except (TypeError, ValueError):
            raise MigrationError(
                "legacy migration metadata is not JSON representable"
            ) from None
        if target is None:
            return report
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".agent-migrate-", dir=target.parent) as staging:
            staged = Database.create(Path(staging) / "workspace.sqlite")
            with staged.connect(write=True) as converted:
                for table, names in columns.items():
                    if table == "sqlite_sequence":
                        converted.execute("DELETE FROM sqlite_sequence")
                    fields = ", ".join(f'"{name}"' for name in names)
                    placeholders = ", ".join("?" for _ in names)
                    converted.executemany(
                        f'INSERT INTO "{table}" ({fields}) VALUES ({placeholders})',
                        _mapped_rows(original, table, names, models, families),
                    )
                if converted.execute("PRAGMA foreign_key_check").fetchall():
                    raise MigrationError(
                        "converted database failed foreign key validation"
                    )
            with staged.connect() as converted:
                actual = {
                    table: _digest(_rows(converted, table, names))
                    for table, names in columns.items()
                }
                if actual != expected:
                    raise MigrationError(
                        "converted database differs from the source mapping"
                    )
                if [row[0] for row in converted.execute("PRAGMA integrity_check")] != [
                    "ok"
                ]:
                    raise MigrationError(
                        "converted database failed SQLite integrity_check"
                    )
            # All connections are closed and WAL checkpointed before exclusive publication.
            os.link(staged.path, target)
        report["verified"] = True
        return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--target", type=Path, help="New output database; omit for dry-run"
    )
    parser.add_argument("--mapping", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        mapping = json.loads(args.mapping.read_text())
        if not isinstance(mapping, dict) or set(mapping) != {"models", "families"}:
            raise ValueError("mapping must contain models and families")
        identities = {
            kind: {key: ProviderIdentity(**value) for key, value in entries.items()}
            for kind, entries in mapping.items()
        }
    except (OSError, ValueError, TypeError, AttributeError):
        parser.error("cannot read a valid provider mapping document")
    try:
        report = migrate_database(args.source, args.target, **identities)
    except (MigrationError, FileExistsError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
