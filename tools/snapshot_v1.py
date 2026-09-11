"""Read-only baseline inventory; never imports or initializes the application.

The output contains schema and data digests, not conversation text or credentials.
Each SQLite inventory uses one read transaction, including committed WAL data.
This is a live inventory, not the stopped, cross-workspace migration backup.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sqlite3
import subprocess
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def repository(path: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args])

    return {
        "path": str(path.resolve()),
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "status": git("status", "--porcelain").decode().splitlines(),
        "tracked_diff_sha256": digest(git("diff", "HEAD", "--binary")),
    }


def routes(source: Path) -> list[dict]:
    tree = ast.parse(source.read_text())
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        dependencies = []
        for value in [*node.args.defaults, *node.args.kw_defaults]:
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "Depends"
                and value.args
            ):
                dependencies.append(ast.unparse(value.args[0]))
        for decorator in node.decorator_list:
            if (
                not isinstance(decorator, ast.Call)
                or not isinstance(decorator.func, ast.Attribute)
                or not isinstance(decorator.func.value, ast.Name)
                or decorator.func.value.id != "app"
                or decorator.func.attr not in {"get", "post", "patch", "delete", "put"}
            ):
                continue
            result.append(
                {
                    "method": decorator.func.attr.upper(),
                    "path": ast.literal_eval(decorator.args[0]),
                    "dependencies": sorted(dependencies),
                    "handler": node.name,
                }
            )
    return sorted(result, key=lambda row: (row["path"], row["method"]))


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def database(path: Path) -> dict:
    with closing(
        sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        schema = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        tables = {}
        for kind, name, _, _ in schema:
            if kind != "table":
                continue
            table = quote_identifier(name)
            columns = connection.execute(f"PRAGMA table_info({table})").fetchall()
            ordering = ", ".join(quote_identifier(column[1]) for column in columns)
            hasher = hashlib.sha256()
            count = 0
            for row in connection.execute(f"SELECT * FROM {table} ORDER BY {ordering}"):
                normalized = [
                    {"blob_hex": v.hex()} if isinstance(v, bytes) else v for v in row
                ]
                hasher.update(
                    json.dumps(
                        normalized, ensure_ascii=True, separators=(",", ":")
                    ).encode()
                )
                hasher.update(b"\n")
                count += 1
            tables[name] = {
                "count": count,
                "sha256": hasher.hexdigest(),
                "columns": columns,
            }
        return {
            "schema": schema,
            "tables": tables,
            "foreign_key_errors": connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall(),
            "relationship_errors": relationships(connection, tables),
        }


def relationships(connection: sqlite3.Connection, tables: dict) -> dict[str, int]:
    """Audit v1 logical links that were never declared as SQLite foreign keys."""
    columns = {
        name: {column[1] for column in table["columns"]}
        for name, table in tables.items()
    }
    errors = {}
    for source, key, target in (
        ("messages", "turn_id", "turns"),
        ("execution_jobs", "turn_id", "turns"),
        ("execution_jobs", "experiment_id", "experiments"),
        ("experiments", "job_id", "execution_jobs"),
        ("conversation_experiments", "turn_id", "turns"),
    ):
        if key not in columns.get(source, set()) or "id" not in columns.get(
            target, set()
        ):
            continue
        errors[f"{source}.{key}.orphaned"] = connection.execute(
            f"SELECT COUNT(*) FROM {source} s LEFT JOIN {target} t ON s.{key} = t.id "
            f"WHERE s.{key} IS NOT NULL AND t.id IS NULL"
        ).fetchone()[0]
        if (
            "conversation_id" in columns[source]
            and "conversation_id" in columns[target]
        ):
            errors[f"{source}.{key}.conversation_mismatch"] = connection.execute(
                f"SELECT COUNT(*) FROM {source} s JOIN {target} t ON s.{key} = t.id "
                "WHERE s.conversation_id != t.conversation_id"
            ).fetchone()[0]
    return errors


def workspace_inventory(root: Path) -> dict:
    registry_path = root / "registry.json"
    raw = registry_path.read_bytes()
    registered = {item["workspace_id"] for item in json.loads(raw)["workspaces"]}
    descriptors = {path.parent.name: path for path in root.glob("*/workspace.json")}
    workspaces = {}
    for identity in sorted(registered | descriptors.keys()):
        descriptor_path = descriptors.get(identity)
        record = {
            "registered": identity in registered,
            "descriptor_present": descriptor_path is not None,
        }
        if descriptor_path is not None:
            descriptor_raw = descriptor_path.read_bytes()
            descriptor = json.loads(descriptor_raw)
            record.update(
                {
                    "descriptor_sha256": digest(descriptor_raw),
                    "storage_kind": descriptor["storage_kind"],
                    "state": descriptor["state"],
                }
            )
            db_path = descriptor_path.parent / "workspace.sqlite"
            record["database"] = database(db_path) if db_path.is_file() else None
        workspaces[identity] = record
    return {
        "root": str(root.resolve()),
        "registry_sha256": digest(raw),
        "workspaces": workspaces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Legacy Agent checkout containing backend/app.py; use the retained checkout after cutover",
    )
    parser.add_argument("--peer", type=Path, action="append", default=[])
    parser.add_argument("--workspaces-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    legacy_app = args.repo / "backend" / "app.py"
    if not legacy_app.is_file():
        parser.error(
            "--repo must point to a retained legacy Agent checkout containing backend/app.py"
        )
    captured_routes = routes(legacy_app)
    if not captured_routes:
        parser.error(
            "legacy backend/app.py has no supported @app routes; no baseline written"
        )
    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "consistency": "per-database read transaction; not a migration backup",
        "repository": repository(args.repo),
        "peers": [repository(peer) for peer in args.peer],
        "routes": captured_routes,
    }
    if args.workspaces_root:
        report["state"] = workspace_inventory(args.workspaces_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A baseline is evidence: rerunning must not silently overwrite it.
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, ensure_ascii=True)
        output.write("\n")
    print(f"Captured {len(report['routes'])} routes in {args.output}")


if __name__ == "__main__":
    main()
