"""Workspace-local managed jobs and their atomic turn-event relationships."""

import sqlite3
import time
from collections.abc import Callable
from typing import Any

from .database import Database
from .turns import _insert_event


def _require_running(
    connection: sqlite3.Connection, conversation_id: str, turn_id: str
) -> None:
    row = connection.execute(
        "SELECT status FROM turns WHERE id = ? AND conversation_id = ?",
        (turn_id, conversation_id),
    ).fetchone()
    if row is None:
        raise KeyError(turn_id)
    if row["status"] != "running":
        raise ValueError("cannot mutate jobs for a terminal turn")


def _job_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload = {
        key: row[key]
        for key in (
            "experiment_id",
            "conversation_id",
            "turn_id",
            "status",
            "experiment_path",
            "job_kind",
            "artifact_path",
            "resource_id",
            "analyzer_resource_id",
        )
    }
    return {"job_id": row["id"], **payload}


class Jobs:
    def __init__(self, database: Database):
        self.database = database

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM execution_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return {**_job_payload(row), "role": row["role"]} if row is not None else None

    def create_simulation(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        role: str,
        job_id: str,
        experiment_id: str,
        experiment_path: str,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        with self.database.connect(write=True) as connection:
            _require_running(connection, conversation_id, turn_id)
            existing = connection.execute(
                "SELECT id FROM experiments WHERE relative_path = ?", (experiment_path,)
            ).fetchone()
            if existing is not None and existing["id"] != experiment_id:
                raise ValueError("experiment path already has a different identity")
            identity = connection.execute(
                "SELECT relative_path FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if identity is not None and identity["relative_path"] != experiment_path:
                raise ValueError(
                    "experiment identity already belongs to a different path"
                )
            connection.execute(
                """INSERT INTO execution_jobs(id, conversation_id, turn_id, role,
                status, job_kind, experiment_id, experiment_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'requested', 'simulation', ?, ?, ?, ?)""",
                (
                    job_id,
                    conversation_id,
                    turn_id,
                    role,
                    experiment_id,
                    experiment_path,
                    now,
                    now,
                ),
            )
            if existing is None:
                connection.execute(
                    """INSERT INTO experiments(id, relative_path, status, origin_kind,
                    job_id, created_at, updated_at) VALUES (?, ?, 'requested', 'managed', ?, ?, ?)""",
                    (experiment_id, experiment_path, job_id, now, now),
                )
            else:
                connection.execute(
                    "UPDATE experiments SET status = 'requested', job_id = ?, updated_at = ? WHERE id = ?",
                    (job_id, now, experiment_id),
                )
            connection.execute(
                """INSERT INTO conversation_experiments(conversation_id, experiment_id, turn_id, relation)
                VALUES (?, ?, ?, 'produced')
                ON CONFLICT(conversation_id, experiment_id, relation)
                DO UPDATE SET turn_id = excluded.turn_id""",
                (conversation_id, experiment_id, turn_id),
            )
            _insert_event(connection, turn_id, event["kind"], event)
        return {
            "job_id": job_id,
            "experiment_id": experiment_id,
            "status": "requested",
            "experiment_path": experiment_path,
            "job_kind": "simulation",
        }

    def create_artifact(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        role: str,
        job_id: str,
        resource_id: str,
        job_kind: str,
        artifact_path: str,
        analyzer_resource_id: str | None,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        if job_kind == "simulation":
            raise ValueError("simulation jobs require an experiment")
        now = time.time()
        with self.database.connect(write=True) as connection:
            _require_running(connection, conversation_id, turn_id)
            connection.execute(
                """INSERT INTO execution_jobs(id, conversation_id, turn_id, role,
                status, job_kind, artifact_path, resource_id, analyzer_resource_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'requested', ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    conversation_id,
                    turn_id,
                    role,
                    job_kind,
                    artifact_path,
                    resource_id,
                    analyzer_resource_id,
                    now,
                    now,
                ),
            )
            _insert_event(connection, turn_id, event["kind"], event)
        return {
            "job_id": job_id,
            "resource_id": resource_id,
            "analyzer_resource_id": analyzer_resource_id,
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "status": "requested",
            "job_kind": job_kind,
            "artifact_path": artifact_path,
        }

    def update(
        self,
        job_id: str,
        *,
        conversation_id: str,
        turn_id: str,
        status: str,
        simulation: bool,
        event: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any] | None:
        """The event callback must be pure; any failure rolls back all mutations."""
        with self.database.connect(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM execution_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if (
                row is None
                or row["conversation_id"] != conversation_id
                or row["turn_id"] != turn_id
                or (row["job_kind"] == "simulation") != simulation
            ):
                return None
            _require_running(connection, conversation_id, turn_id)
            now = time.time()
            connection.execute(
                "UPDATE execution_jobs SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, job_id),
            )
            if row["experiment_id"]:
                connection.execute(
                    "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ? AND job_id = ?",
                    (status, now, row["experiment_id"], job_id),
                )
            result = _job_payload(row)
            result["status"] = status
            payload = event(result.copy())
            _insert_event(connection, turn_id, payload["kind"], payload)
        return result

    def experiment_by_path(self, path: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE relative_path = ?", (path,)
            ).fetchone()
        return dict(row) if row is not None else None

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def artifact_by_resource(self, resource_id: str) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT id AS job_id, resource_id, analyzer_resource_id, conversation_id,
                turn_id, role, status, job_kind, created_at, updated_at FROM execution_jobs
                WHERE resource_id = ? AND job_kind != 'simulation'""",
                (resource_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_artifacts(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT j.id AS job_id, j.conversation_id, c.title AS conversation_title,
                j.turn_id, j.status, j.job_kind, j.artifact_path, j.resource_id,
                j.analyzer_resource_id, j.created_at, j.updated_at
                FROM execution_jobs j JOIN conversations c ON c.id = j.conversation_id
                WHERE j.job_kind != 'simulation' AND j.resource_id IS NOT NULL
                ORDER BY j.updated_at DESC, j.id"""
            ).fetchall()
        return [dict(row) for row in rows]

    def list_experiments(
        self, conversation_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT e.*, ce.conversation_id, ce.turn_id, ce.relation
                FROM experiments e LEFT JOIN conversation_experiments ce ON ce.experiment_id = e.id
                WHERE (? IS NULL OR ce.conversation_id = ?)
                ORDER BY e.updated_at DESC, e.id, ce.conversation_id, ce.relation""",
                (conversation_id, conversation_id),
            ).fetchall()
        return [dict(row) for row in rows]
