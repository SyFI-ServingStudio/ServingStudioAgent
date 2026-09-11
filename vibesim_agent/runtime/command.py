"""Docker exec arguments assembled from an explicit execution environment."""

from collections.abc import Mapping
from dataclasses import dataclass

from ..settings import AgentSettings, ContainerSettings


@dataclass(frozen=True)
class ExecutionEnvironment:
    container: ContainerSettings
    agent: AgentSettings
    lock_sha: str
    managed_context: str

    def prefix(
        self,
        container_name: str,
        *,
        environment: Mapping[str, str],
        inherited: tuple[str, ...] = (),
    ) -> list[str]:
        settings = self.container
        variables = {
            "HOME": str(settings.home),
            **environment,
            "USER": settings.user,
            "LOGNAME": settings.user,
            "UV_PROJECT_ENVIRONMENT": str(settings.uv_project_environment),
            "UV_CACHE_DIR": str(settings.uv_cache_dir),
            "VIBESIM_EXPECTED_LOCK_SHA": self.lock_sha,
            "DG_USE_LOCAL_VERSION": str(int(settings.dg_use_local_version)),
            "VIBESIM_RUNNER_GPUS": settings.gpus,
            "ANALYZER_MCP_SOURCE": self.agent.analyzer_source,
            "ANALYZER_MCP_BASE_URL": self.agent.analyzer_base_url,
            "VIBESIM_MANAGED_RUN_CONTEXT": self.managed_context,
            "VIBESIM_MANAGED_JOB_CONTEXT": self.managed_context,
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        }
        if set(inherited).intersection(variables):
            raise ValueError(
                "inherited environment conflicts with explicit runtime values"
            )
        return [
            "docker",
            "exec",
            "-i",
            "-u",
            f"{settings.uid}:{settings.gid}",
            *(
                arg
                for name, value in variables.items()
                for arg in ("-e", f"{name}={value}")
            ),
            *(arg for name in inherited for arg in ("-e", name)),
            "-w",
            "/workspace",
            container_name,
        ]
