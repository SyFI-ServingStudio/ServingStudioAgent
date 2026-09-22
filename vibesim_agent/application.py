"""Explicit service assembly over existing workspace state and provider factories."""

import hashlib
import logging
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI

from .api.citations import citation_router
from .api.files import file_router
from .api.jobs import job_router
from .api.tools import tools_router
from .api.workspaces import workspace_router
from .domain.conversations import RoleRuntime
from .domain.roles import Role
from .main import create_app
from .prompts.render import Prompts
from .providers.base import AgentRequest
from .providers.registry import ProviderRegistry
from .runtime.command import ExecutionEnvironment
from .runtime.container import ContainerManager
from .runtime.git import GitRunner
from .runtime.homes import role_home
from .runtime.host import host_workspace_roots
from .runtime.invocation import InvocationHome
from .runtime.mounts import PROMPTS_TARGET, managed_context_target
from .runtime.workspace import WorkspaceSnapshot
from .runtime.worktree import WorktreeProvisioner
from .services.artifacts import ArtifactService
from .services.capabilities import CapabilityRegistry, ManagedContext
from .services.citations import CitationService
from .services.conversation import ConversationService
from .services.driver import ConversationDriver
from .services.eval import EvalService
from .services.jobs import JobService
from .services.name_generator import NameGenerator
from .services.naming import NamingService
from .services.recovery import RecoveryService
from .services.runtime import (
    HostRuntime,
    ProviderRuntime,
    RuntimeService,
    WorkspaceRuntime,
)
from .services.turn import TurnService, TurnStorage
from .services.workspace import WorkspaceService
from .settings import Settings
from .storage.conversations import Conversations
from .storage.database import Database
from .storage.jobs import Jobs
from .storage.ownership import WorkspaceOwnership
from .storage.registry import WorkspaceRegistry
from .storage.sessions import Sessions
from .storage.turns import Turns


@dataclass(frozen=True)
class ProviderSetup:
    registry: ProviderRegistry
    runtimes: Mapping[str, ProviderRuntime]
    defaults: Mapping[Role, RoleRuntime]
    model_aliases: Mapping[str, str] = field(default_factory=dict)
    docker_environment: Mapping[str, str] | None = None


def build_application(
    settings: Settings,
    *,
    providers: Callable[
        [Callable[[AgentRequest], InvocationHome], ExecutionEnvironment, Prompts],
        ProviderSetup,
    ],
    before_call: Callable[[AgentRequest], Awaitable[None]] | None = None,
    prompts_directory: Path,
    mcp_directory: Path,
    managed_context: str,
    namespace: str,
    submodules: tuple[Path, ...],
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    workspace_environment: Mapping[str, str] | None = None,
    ownership: WorkspaceOwnership | None = None,
) -> FastAPI:
    """Assemble explicitly; no workspace creation or runtime migration occurs."""
    container_root = str(settings.container.home / ".vibesim-agent")
    context_target = managed_context_target(
        managed_context, settings.container, container_root
    )
    workspaces = WorkspaceRegistry(settings.agent.workspaces_root)
    artifacts = ArtifactService(workspaces)
    worktree_root = settings.agent.worktree_root
    workspace_service = WorkspaceService(
        workspaces,
        WorkspaceSnapshot(
            settings.agent.main_dir,
            process_environment=workspace_environment or {},
        ),
        # Opt-in: provisioning writes real branches into the user's checkout, so
        # a deployment that has not chosen a location does not get the feature.
        worktrees=(
            # Real `subprocess.run`, like the snapshot beside it. `run` here is
            # the Docker boundary; sharing it would let a stubbed container
            # runner silently stub Git too.
            WorktreeProvisioner(
                settings.agent.main_dir,
                process_environment=workspace_environment or {},
            )
            if worktree_root is not None
            else None
        ),
        worktree_root=worktree_root,
    )
    stores = {}

    def storage(workspace_id):
        workspaces.get(workspace_id)
        if workspace_id not in stores:
            database = Database(workspaces.database_path(workspace_id))
            with database.connect():
                pass
            stores[workspace_id] = TurnStorage(
                Conversations(database), Sessions(database), Turns(database)
            )
        return stores[workspace_id]

    for workspace in workspaces.list():
        storage(workspace["workspace_id"])
    lock_sha = hashlib.sha256(
        (settings.agent.main_dir / "uv.lock").read_bytes()
    ).hexdigest()
    environment = ExecutionEnvironment(
        settings.container, settings.agent, lock_sha, managed_context
    )
    prompts = Prompts.prepare(prompts_directory)

    def home(request):
        return role_home(
            workspaces.conversation_runtime_path(
                request.workspace_id, request.conversation_id
            ),
            container_root,
            request.role,
            request.selection.session_scope,
        )

    configured = providers(home, environment, prompts)

    def context_directory(workspace_id, conversation_id):
        root = workspaces.conversation_runtime_path(workspace_id, conversation_id)
        directory = root / "managed"
        if directory.is_symlink() or not directory.resolve().is_relative_to(
            root.resolve()
        ):
            raise ValueError("managed context directory escapes conversation runtime")
        return directory

    capabilities = CapabilityRegistry(clock=time.time)
    context = ManagedContext(
        capabilities,
        lambda workspace_id, conversation_id: (
            context_directory(workspace_id, conversation_id) / context_target.name
        ),
    )

    async def prepare_call(request):
        context.write(request)
        if before_call is not None:
            await before_call(request)

    def finish_turn(request):
        capabilities.revoke_turn(request.workspace_id, request.turn_id)
        context.remove(request.workspace_id, request.conversation_id)

    def workspace_runtime(workspace_id: str) -> WorkspaceRuntime:
        storage_kind = workspaces.get(workspace_id)["storage_kind"]
        return WorkspaceRuntime(
            workspaces.repo_path(workspace_id),
            workspaces.workspace_dir(workspace_id),
            # The mount tuple exists to fill the empty gitlink directories a
            # managed copy leaves behind. A real git tree already has its
            # submodules checked out, and mounting over them would hide work.
            () if storage_kind == "external" else submodules,
            storage_kind=storage_kind,
        )

    runtime = RuntimeService(
        workspace=workspace_runtime,
        providers=configured.registry,
        runtimes=configured.runtimes,
        containers=ContainerManager(
            environment, run=run, process_environment=configured.docker_environment
        ),
        prompts=prompts,
        mcp=mcp_directory,
        container_root=container_root,
        namespace=namespace,
        conversation_path=workspaces.conversation_runtime_path,
        context_directory=context_directory,
        prepare_workspace=workspace_service.prepare,
        host=HostRuntime(
            # `host.docker.internal` resolves only inside a container. Both
            # services already bind a host interface, so what a host turn needs
            # is the same address under a name this machine can resolve --
            # configured explicitly rather than rewritten at runtime.
            analyzer_base_url=(
                settings.agent.host_analyzer_base_url
                or settings.agent.analyzer_base_url
            ),
            managed_backend_url=(
                settings.agent.host_managed_backend_url
                or settings.agent.managed_backend_url
            ),
            # The image's MCP virtualenv does not exist here; the Agent's own
            # interpreter already has the same pinned `mcp` package.
            mcp_python=sys.executable,
            mcp_server=(
                settings.agent.repo_root
                / "vibesim_agent/analyzer_evidence_mcp/server.py"
            ),
            git=GitRunner(workspace_environment or {}),
            workspace_roots=host_workspace_roots(workspace_environment or {}),
        ),
    )
    driver = ConversationDriver(
        configured.registry,
        prompts,
        prepare=runtime.prepare,
        before_call=prepare_call,
        schema_directory=Path(str(PROMPTS_TARGET)),
        after_turn=finish_turn,
    )
    naming = NamingService(
        workspaces,
        lambda workspace_id: storage(workspace_id).conversations,
        NameGenerator(
            api_key=settings.secrets.get("OPENROUTER_API_KEY"),
            model=settings.agent.naming_model,
            base_url=settings.agent.naming_base_url,
            timeout=settings.agent.naming_timeout,
            prompts=prompts,
        ),
    )
    turns = TurnService(
        storage,
        driver,
        safe_interrupt_timeout=settings.agent.safe_interrupt_timeout,
        logger=logging.getLogger("vibesim_agent.turn"),
        fingerprint=lambda request: prompts.fingerprint(
            mode=request.mode, autonomous=request.autonomous, runtimes=request.runtimes
        ),
        naming=naming,
    )
    conversations = ConversationService(
        storage,
        providers=configured.registry,
        default_runtimes=configured.defaults,
        fingerprint=prompts.fingerprint,
        model_aliases=configured.model_aliases,
        touch_workspace=lambda workspace_id: workspaces.update(
            workspace_id, touch=True
        ),
        workspaces=workspaces,
    )
    # Resolve default capability settings before accepting requests.
    conversations.browser_runtimes({})
    evaluations = EvalService(
        workspace_service,
        conversations,
        turns,
        release=runtime.release,
    )
    recovery = RecoveryService(
        workspaces,
        storage,
        capabilities=capabilities,
        context=context,
        release=runtime.release,
        ownership=ownership,
    )

    async def shutdown():
        naming.stop_accepting()
        try:
            await evaluations.close()
        finally:
            try:
                await naming.close()
            finally:
                recovery.close()

    app = create_app(
        turns,
        conversations=conversations,
        cleanup_conversation=runtime.cleanup,
        prepare_workspace=workspace_service.prepare,
        startup=recovery.recover,
        shutdown=shutdown,
    )
    app.include_router(workspace_router(workspace_service))
    app.include_router(file_router(artifacts))
    app.include_router(
        tools_router(
            turns,
            conversations,
            token=settings.agent.api_token,
            skill_document=settings.agent.repo_root / "SKILL.md",
            cleanup_conversation=runtime.cleanup,
            prepare_workspace=workspace_service.prepare,
            workspaces=workspace_service,
            evaluations=evaluations,
            artifacts=artifacts,
        )
    )
    jobs = JobService(
        lambda workspace_id: Jobs(storage(workspace_id).turns.database),
        workspaces,
        turns,
    )
    app.include_router(job_router(jobs, capabilities))
    app.include_router(
        citation_router(CitationService(turns, jobs.storage), capabilities)
    )
    app.state.jobs = jobs
    app.state.turns = turns
    app.state.runtime = runtime
    app.state.workspaces = workspaces
    app.state.workspace_service = workspace_service
    app.state.evaluations = evaluations
    app.state.artifacts = artifacts
    app.state.naming = naming
    app.state.recovery = recovery
    app.state.capabilities = capabilities
    app.state.managed_context = context
    return app
