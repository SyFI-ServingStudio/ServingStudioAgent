"""Managed job ownership and paths; Analyzer remains the result authority."""

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from ..storage.jobs import Jobs
from ..storage.registry import WorkspaceRegistry
from .capabilities import Capability
from .turn import TurnService


class JobConflict(ValueError):
    pass


RESOURCE_PREFIXES = {
    "timing_predict": "p_",
    "kernel_profile": "kp_",
    "kernel_measure": "km_",
}
SIMULATION_EVENTS = {
    "running": "simulation.running",
    "analysis_running": "analysis.running",
    "ready": "experiment.ready",
    "failed": "experiment.failed",
    "interrupted": "experiment.interrupted",
}


class JobService:
    def __init__(
        self,
        storage: Callable[[str], Jobs],
        workspaces: WorkspaceRegistry,
        turns: TurnService,
    ):
        self.storage = storage
        self.workspaces = workspaces
        self.turns = turns

    def _owner(self, capability: Capability) -> Jobs:
        store = self.turns.storage(capability.workspace_id)
        if store.conversations.get(capability.conversation_id) is None:
            raise KeyError("managed conversation not found")
        turn = store.turns.get(capability.conversation_id, capability.turn_id)
        if turn is None:
            raise KeyError("managed turn not found")
        if turn["status"] != "running":
            raise JobConflict("managed turn has already ended")
        return self.storage(capability.workspace_id)

    def _root(
        self, capability: Capability, requested_root: str, label: str
    ) -> tuple[Path, str, str]:
        if not requested_root.strip():
            raise ValueError(f"{label} must not be empty")
        repo = self.workspaces.repo_path(capability.workspace_id).resolve()
        logs = self.workspaces.logs_path(capability.workspace_id).resolve()
        requested = Path(requested_root)
        if requested.is_relative_to("/workspace"):
            host = repo / requested.relative_to("/workspace")
        else:
            host = requested if requested.is_absolute() else repo / requested
        resolved = host.resolve()
        if resolved == logs:
            raise ValueError(
                f"{label} must name a folder below the workspace logs root"
            )
        if not resolved.is_relative_to(logs):
            raise PermissionError(f"{label} is outside the workspace logs root")
        return resolved, resolved.relative_to(logs).as_posix(), requested.as_posix()

    @staticmethod
    def _metadata(path: Path) -> dict | None:
        if path.is_symlink():
            raise JobConflict("existing experiment metadata must not be a symlink")
        if not path.exists():
            return None
        try:
            metadata = json.loads(path.read_text("utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise JobConflict(
                "existing experiment metadata could not be read"
            ) from error
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != 1
            or not isinstance(metadata.get("experiment_id"), str)
            or not metadata["experiment_id"]
        ):
            raise JobConflict("existing experiment metadata is incompatible")
        return metadata

    def _publish_metadata(
        self, path: Path, metadata: dict, *, expected: dict | None = None
    ) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Publish a complete inode without replacing another registration's ID.
        with TemporaryDirectory(
            prefix=".experiment-meta-", dir=path.parent
        ) as directory:
            temporary = Path(directory) / "metadata.json"
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            if expected is not None:
                if self._metadata(path) != expected:
                    raise JobConflict("experiment metadata changed during registration")
                temporary.replace(path)
            else:
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    existing = self._metadata(path)
                    if existing is None:
                        raise JobConflict(
                            "experiment metadata changed during registration"
                        ) from None
                    return existing
        return metadata

    def _recover_metadata(
        self,
        jobs: Jobs,
        path: Path,
        metadata: dict,
        workspace_id: str,
        relative_path: str,
    ) -> dict:
        pending = metadata.get("agent_registration")
        if "agent_registration" not in metadata:
            return metadata
        if (
            not isinstance(pending, dict)
            or type(pending.get("version")) is not int
            or pending["version"] != 1
            or set(pending) != {"version", "origin"}
            or metadata.get("origin") != {"kind": "managed"}
        ):
            raise JobConflict("pending experiment registration is incompatible")
        pending = pending["origin"]
        if (
            not isinstance(pending, dict)
            or pending.get("workspace_id") != workspace_id
            or pending.get("kind") != "managed"
            or any(
                not isinstance(pending.get(key), str) or not pending[key]
                for key in ("job_id", "conversation_id", "turn_id", "role")
            )
        ):
            raise JobConflict("pending experiment registration is incompatible")
        job = jobs.get(pending["job_id"])
        if job is None:
            return metadata
        if (
            job["experiment_id"] != metadata["experiment_id"]
            or job["experiment_path"] != relative_path
            or job["job_kind"] != "simulation"
            or any(
                job[key] != pending[key]
                for key in ("conversation_id", "turn_id", "role")
            )
        ):
            raise JobConflict(
                "pending experiment registration disagrees with workspace database"
            )
        return self._finalize_metadata(path, metadata)

    def _finalize_metadata(self, path: Path, metadata: dict) -> dict:
        final = {
            key: value for key, value in metadata.items() if key != "agent_registration"
        }
        final["origin"] = metadata["agent_registration"]["origin"]
        return self._publish_metadata(path, final, expected=metadata)

    @staticmethod
    def _identity(capability: Capability, job_id: str) -> dict:
        return {
            "workspaceId": capability.workspace_id,
            "conversationId": capability.conversation_id,
            "turnId": capability.turn_id,
            "jobId": job_id,
        }

    def _notify(self, capability: Capability) -> None:
        handle = self.turns.current(capability.workspace_id, capability.conversation_id)
        if (
            handle is not None
            and handle.request.turn_id == capability.turn_id
            and not handle.finished
        ):
            handle.notify()

    def register(
        self,
        capability: Capability,
        *,
        job_kind: str,
        artifact_root: str,
        analyzer_resource_id: str | None = None,
        run_count: int = 1,
        axes: list[str] | None = None,
    ) -> dict:
        if job_kind == "simulation":
            if analyzer_resource_id is not None:
                raise ValueError(
                    "simulation assigns its experiment identity at registration"
                )
            return self.register_run(
                capability,
                experiment_root=artifact_root,
                run_count=run_count,
                axes=axes or [],
            )
        if run_count != 1 or axes:
            raise ValueError("runCount and axes apply only to simulation")
        return self.register_artifact(
            capability,
            job_kind=job_kind,
            artifact_root=artifact_root,
            analyzer_resource_id=analyzer_resource_id,
        )

    def update_registered(
        self, capability: Capability, job_id: str, status: str
    ) -> dict:
        job = self._owner(capability).get(job_id)
        if (
            job is None
            or job["conversation_id"] != capability.conversation_id
            or job["turn_id"] != capability.turn_id
        ):
            raise KeyError("managed job not found")
        return self.update(
            capability, job_id, status, simulation=job["job_kind"] == "simulation"
        )

    def register_run(
        self,
        capability: Capability,
        *,
        experiment_root: str,
        run_count: int,
        axes: list[str],
    ) -> dict:
        jobs = self._owner(capability)
        root, relative, approved = self._root(
            capability, experiment_root, "experimentRoot"
        )
        path = root / "experiment.meta.json"
        metadata = self._metadata(path)
        if metadata is not None:
            metadata = self._recover_metadata(
                jobs, path, metadata, capability.workspace_id, relative
            )
        existing = jobs.experiment_by_path(relative)
        if (
            metadata is not None
            and existing is not None
            and metadata["experiment_id"] != existing["id"]
        ):
            raise JobConflict(
                "filesystem and workspace database disagree on experiment identity"
            )
        job_id = "j_" + uuid4().hex
        origin = {
            "kind": "managed",
            "workspace_id": capability.workspace_id,
            "conversation_id": capability.conversation_id,
            "turn_id": capability.turn_id,
            "job_id": job_id,
            "role": capability.role,
        }
        if metadata is None:
            metadata = self._publish_metadata(
                path,
                {
                    "schema_version": 1,
                    "experiment_id": existing["id"] if existing else "e_" + uuid4().hex,
                    "origin": {"kind": "managed"},
                    "agent_registration": {"version": 1, "origin": origin},
                },
            )
            metadata = self._recover_metadata(
                jobs, path, metadata, capability.workspace_id, relative
            )
        if (
            "agent_registration" in metadata
            and metadata["agent_registration"]["origin"] != origin
        ):
            metadata = self._publish_metadata(
                path,
                {**metadata, "agent_registration": {"version": 1, "origin": origin}},
                expected=metadata,
            )
        experiment_id = metadata["experiment_id"]
        event = {
            "kind": "simulation.requested",
            **self._identity(capability, job_id),
            "experimentId": experiment_id,
            "experimentPath": relative,
            "runCount": run_count,
            "axes": axes,
        }
        try:
            jobs.create_simulation(
                conversation_id=capability.conversation_id,
                turn_id=capability.turn_id,
                role=capability.role,
                job_id=job_id,
                experiment_id=experiment_id,
                experiment_path=relative,
                event=event,
            )
        except ValueError as error:
            raise JobConflict(str(error)) from error
        self._notify(capability)
        if "agent_registration" in metadata:
            self._finalize_metadata(path, metadata)
        return {
            "schemaVersion": 1,
            **self._identity(capability, job_id),
            "experimentId": experiment_id,
            "approvedRoot": approved,
        }

    def register_artifact(
        self,
        capability: Capability,
        *,
        job_kind: str,
        artifact_root: str,
        analyzer_resource_id: str | None,
    ) -> dict:
        prefix = RESOURCE_PREFIXES.get(job_kind)
        if prefix is None:
            raise ValueError("unsupported managed job kind")
        if (
            not isinstance(analyzer_resource_id, str)
            or re.fullmatch(
                re.escape(prefix) + r"[a-z0-9_]{1,64}", analyzer_resource_id
            )
            is None
        ):
            raise ValueError(f"{job_kind} requires a valid analyzerResourceId")
        jobs = self._owner(capability)
        _, relative, approved = self._root(capability, artifact_root, "artifactRoot")
        job_id, resource_id = "j_" + uuid4().hex, "jr_" + uuid4().hex
        event = {
            "kind": "job.requested",
            **self._identity(capability, job_id),
            "jobKind": job_kind,
            "resourceId": resource_id,
            "analyzerResourceId": analyzer_resource_id,
            "status": "requested",
        }
        try:
            jobs.create_artifact(
                conversation_id=capability.conversation_id,
                turn_id=capability.turn_id,
                role=capability.role,
                job_id=job_id,
                resource_id=resource_id,
                job_kind=job_kind,
                artifact_path=relative,
                analyzer_resource_id=analyzer_resource_id,
                event=event,
            )
        except ValueError as error:
            raise JobConflict(str(error)) from error
        self._notify(capability)
        return {
            "schemaVersion": 1,
            **self._identity(capability, job_id),
            "resourceId": resource_id,
            "analyzerResourceId": analyzer_resource_id,
            "approvedRoot": approved,
        }

    def update(
        self, capability: Capability, job_id: str, status: str, *, simulation: bool
    ) -> dict:
        if status not in SIMULATION_EVENTS:
            raise ValueError(
                "unsupported managed-run status"
                if simulation
                else "unsupported managed-job status"
            )
        jobs = self._owner(capability)

        def event(job):
            payload = {**self._identity(capability, job_id), "status": status}
            if simulation:
                return {
                    "kind": SIMULATION_EVENTS[status],
                    **payload,
                    "experimentId": job["experiment_id"],
                    "experimentPath": job["experiment_path"],
                }
            return {
                "kind": "job." + status,
                **payload,
                "jobKind": job["job_kind"],
                "resourceId": job["resource_id"],
                "analyzerResourceId": job["analyzer_resource_id"],
            }

        try:
            job = jobs.update(
                job_id,
                conversation_id=capability.conversation_id,
                turn_id=capability.turn_id,
                status=status,
                simulation=simulation,
                event=event,
            )
        except ValueError as error:
            raise JobConflict(str(error)) from error
        if job is None:
            raise KeyError("managed job not found")
        self._notify(capability)
        return event(job)

    def list_jobs(self) -> list[dict]:
        fields = {
            "job_id",
            "conversation_id",
            "conversation_title",
            "turn_id",
            "status",
            "job_kind",
            "resource_id",
            "analyzer_resource_id",
            "created_at",
            "updated_at",
        }
        jobs = [
            {
                "workspace_id": workspace["workspace_id"],
                **{key: value for key, value in job.items() if key in fields},
            }
            for workspace in self.workspaces.list()
            for job in self.storage(workspace["workspace_id"]).list_artifacts()
        ]
        return sorted(
            jobs,
            key=lambda job: (-job["updated_at"], job["workspace_id"], job["job_id"]),
        )

    def resource(self, workspace_id: str, resource_id: str) -> dict:
        job = self.storage(workspace_id).artifact_by_resource(resource_id)
        if job is None:
            raise KeyError("managed job resource not found")
        return {
            "schemaVersion": 1,
            "workspaceId": workspace_id,
            "jobId": job["job_id"],
            "resourceId": job["resource_id"],
            "analyzerResourceId": job["analyzer_resource_id"],
            "jobKind": job["job_kind"],
            "status": job["status"],
        }

    def experiments(self, workspace_id: str, conversation_id: str) -> dict:
        if self.turns.storage(workspace_id).conversations.get(conversation_id) is None:
            raise KeyError("conversation not found")
        return {
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
            "experiments": self.storage(workspace_id).list_experiments(
                conversation_id=conversation_id
            ),
        }
