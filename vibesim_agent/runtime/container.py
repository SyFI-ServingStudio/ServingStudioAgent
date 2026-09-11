"""Owned Docker lifecycle with explicit configuration and readiness checks."""

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from uuid import uuid4

from .bootstrap import bootstrap_script, readiness_script
from .command import ExecutionEnvironment
from .mounts import Mount

OWNER_LABEL = "vibesim.agent.owner"
CONFIG_LABEL = "vibesim.agent.configuration"
CREATION_LABEL = "vibesim.agent.creation"


@dataclass(frozen=True)
class ContainerSpec:
    name: str
    owner: str
    mounts: tuple[Mount, ...]
    role_homes: tuple[str, ...]
    binaries: tuple[str, ...]
    agent_prompt: str
    session_scopes: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", self.name):
            raise ValueError("invalid container name")
        if not self.owner:
            raise ValueError("container owner is required")
        targets = [mount.target for mount in self.mounts]
        if len(set(targets)) != len(targets):
            raise ValueError("duplicate container mount targets")


class ContainerManager:
    def __init__(
        self,
        environment: ExecutionEnvironment,
        *,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        process_environment: Mapping[str, str] | None = None,
    ):
        self.environment = environment
        self.run = run
        self.process_environment = (
            dict(process_environment) if process_environment is not None else None
        )

    def _run(self, arguments: list[str], *, timeout: int = 30, check=True):
        result = self.run(
            arguments,
            # Management exec uses -i but must not read a background job's terminal.
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=self.process_environment,
        )
        if check and result.returncode:
            # Docker diagnostics can contain runtime environment values.
            raise RuntimeError(
                f"Docker {arguments[1]} failed with status {result.returncode}"
            )
        return result

    def _inspect(self, name: str) -> dict | None:
        result = self._run(["docker", "container", "inspect", name], check=False)
        if result.returncode:
            if (
                "No such container:" in result.stderr
                or "No such object:" in result.stderr
            ):
                return None
            raise RuntimeError("Docker container inspection failed")
        objects = json.loads(result.stdout)
        if (
            not isinstance(objects, list)
            or len(objects) != 1
            or not isinstance(objects[0], dict)
        ):
            raise RuntimeError("Docker returned invalid container inspection")
        return objects[0]

    def _fingerprint(self, spec: ContainerSpec, image_id: str) -> str:
        payload = {
            "image_id": image_id,
            "container": self.environment.container.model_dump(mode="json"),
            "lock_sha": self.environment.lock_sha,
            "analyzer_source": self.environment.agent.analyzer_source,
            "analyzer_base_url": self.environment.agent.analyzer_base_url,
            "managed_context": self.environment.managed_context,
            "mounts": [
                (str(m.source), str(m.target), m.read_only) for m in spec.mounts
            ],
            "role_homes": spec.role_homes,
            "binaries": spec.binaries,
            "agent_prompt": spec.agent_prompt,
            "session_scopes": spec.session_scopes,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _exec(self, container: str, script: str, *, timeout=30, check=True):
        variables = {}
        if self.environment.container.hf_home is not None:
            variables["HF_HOME"] = "/model"
        return self._run(
            [
                *self.environment.prefix(container, environment=variables),
                "bash",
                "-lc",
                script,
            ],
            timeout=timeout,
            check=check,
        )

    def _create_command(
        self, spec: ContainerSpec, fingerprint: str, image_id: str, creation: str
    ):
        settings = self.environment.container
        command = [
            "docker",
            "create",
            "--init",
            "--name",
            spec.name,
            "--label",
            f"{OWNER_LABEL}={spec.owner}",
            "--label",
            f"{CONFIG_LABEL}={fingerprint}",
            "--label",
            f"{CREATION_LABEL}={creation}",
            "--add-host",
            "host.docker.internal:host-gateway",
            "--user",
            f"{settings.uid}:{settings.gid}",
        ]
        for mount in spec.mounts:
            # Revalidate sources in case provisioning removed a path since the spec was built.
            command.extend(
                Mount(mount.source, mount.target, mount.read_only).docker_args()
            )
        variables = {
            "PYTHONUNBUFFERED": "1",
            "HOME": str(settings.home),
            "USER": settings.user,
            "LOGNAME": settings.user,
            "UV_PROJECT_ENVIRONMENT": str(settings.uv_project_environment),
            "UV_CACHE_DIR": str(settings.uv_cache_dir),
            "VIBESIM_EXPECTED_LOCK_SHA": self.environment.lock_sha,
            "DG_USE_LOCAL_VERSION": str(int(settings.dg_use_local_version)),
            "VIBESIM_RUNNER_GPUS": settings.gpus,
            "ANALYZER_MCP_SOURCE": self.environment.agent.analyzer_source,
            "ANALYZER_MCP_BASE_URL": self.environment.agent.analyzer_base_url,
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        }
        if settings.hf_home is not None:
            variables["HF_HOME"] = "/model"
        for key, value in variables.items():
            command.extend(["-e", f"{key}={value}"])
        if settings.gpus:
            command.extend(["--gpus", settings.gpus])
        command.extend(["-w", "/workspace", image_id, "sleep", "infinity"])
        return command

    def ensure(self, spec: ContainerSpec) -> str:
        image = self._run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                self.environment.container.image,
            ]
        )
        image_id = image.stdout.strip()
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image_id):
            raise RuntimeError("Docker returned invalid image identity")
        fingerprint = self._fingerprint(spec, image_id)
        scripts = {
            "fingerprint": fingerprint,
            "binaries": spec.binaries,
            "role_homes": spec.role_homes,
            "agent_prompt": spec.agent_prompt,
        }
        creation = uuid4().hex
        create = self._create_command(spec, fingerprint, image_id, creation)
        current = self._inspect(spec.name)
        if current is not None:
            labels = current.get("Config", {}).get("Labels") or {}
            if labels.get(OWNER_LABEL) != spec.owner:
                raise RuntimeError("container name belongs to another owner")
            container_id = current["Id"]
            if (
                current.get("State", {}).get("Running")
                and current.get("Image") == image_id
                and labels.get(CONFIG_LABEL) == fingerprint
            ):
                ready = self._exec(
                    container_id,
                    readiness_script(self.environment, **scripts),
                    check=False,
                )
                if ready.returncode == 0:
                    return container_id
            self._run(["docker", "rm", "-f", container_id])
        created = None
        try:
            identity = self._run(create, timeout=180).stdout.strip()
            if not re.fullmatch(r"[a-f0-9]{64}", identity):
                raise RuntimeError("Docker returned invalid new container identity")
            created = identity
            self._run(["docker", "start", created], timeout=180)
            self._exec(
                created, bootstrap_script(self.environment, **scripts), timeout=300
            )
        except BaseException as startup_error:
            try:
                if created is None:
                    candidate = self._inspect(spec.name)
                    labels = (candidate or {}).get("Config", {}).get("Labels") or {}
                    if (
                        labels.get(OWNER_LABEL) == spec.owner
                        and labels.get(CREATION_LABEL) == creation
                    ):
                        created = candidate["Id"]
                if created is not None:
                    self._run(["docker", "rm", "-f", created])
            except Exception as cleanup_error:  # noqa: BLE001 - retain the original startup failure
                startup_error.add_note(
                    f"Container cleanup also failed ({type(cleanup_error).__name__})"
                )
            raise
        return created

    def remove(self, name: str, *, owner: str) -> None:
        current = self._inspect(name)
        if current is None:
            return
        labels = current.get("Config", {}).get("Labels") or {}
        if labels.get(OWNER_LABEL) != owner:
            raise RuntimeError("container name belongs to another owner")
        self._run(["docker", "rm", "-f", current["Id"]])
