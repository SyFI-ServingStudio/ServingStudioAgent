"""Explicit runtime inputs for new-package tests, independent of legacy config."""

from pathlib import Path

from vibesim_agent.domain.roles import Role
from vibesim_agent.providers.base import AgentRequest, Model, Selection
from vibesim_agent.runtime.command import ExecutionEnvironment
from vibesim_agent.settings import AgentSettings, ContainerSettings


def execution_environment() -> ExecutionEnvironment:
    return ExecutionEnvironment(
        ContainerSettings(
            image="test",
            uid=1000,
            gid=1000,
            user="runner",
            home=Path("/home/runner"),
            gpus="all",
            uv_project_environment=Path("/opt/vibesim-venv"),
            uv_cache_dir=Path("/opt/vibesim-uv-cache"),
            dg_use_local_version=False,
        ),
        AgentSettings(
            repo_root=Path("/agent"),
            main_dir=Path("/repo"),
            workspaces_root=Path("/state"),
            analyzer_source="host",
            analyzer_base_url="http://host.docker.internal:63048",
        ),
        "test-main-lock-sha",
        "/opt/vibesim/managed/context.json",
    )


def agent_request() -> AgentRequest:
    model = Model("gpt-5.6-sol", "GPT", ("high",), "high", ("default", "fast"))
    return AgentRequest(
        "w",
        "c",
        "t",
        Role.ASSISTANT,
        "text",
        "container",
        Selection("gpt", model, "high", "fast", "scope"),
        output_schema=Path("/schema.json"),
    )
