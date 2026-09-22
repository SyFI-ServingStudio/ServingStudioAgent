"""Prepare declared provider homes and one owned container for a turn."""

import asyncio
import hashlib
import json
import logging
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..domain.errors import ProviderUnavailable
from ..domain.identifiers import validate_conversation_id
from ..domain.turns import TurnInput
from ..prompts.render import Prompts
from ..providers.base import AgentRequest
from ..providers.registry import ProviderRegistry
from ..runtime.container import ContainerManager, ContainerSpec
from ..runtime.execution import DockerExecution, Execution, HostExecution
from ..runtime.git import GitRunner
from ..runtime.homes import role_home
from ..runtime.host import check_host_binaries, reap_process_groups
from ..runtime.invocation import InvocationHome, RoleContext
from ..runtime.mounts import (
    PROMPTS_TARGET,
    WORKSPACE_TARGET,
    Mount,
    managed_context_target,
    workspace_mounts,
)
from ..runtime.permissions import host_permissions


@dataclass(frozen=True)
class WorkspaceRuntime:
    repo: Path
    state: Path
    submodules: tuple[Path, ...] = ()
    # The execution mode, spelled the way the registry already spells it:
    # `managed` is a copy of the tracked files and runs in a container,
    # `external` is a real git tree and runs here.
    storage_kind: str = "managed"


@dataclass(frozen=True)
class ProviderRuntime:
    prepare: Callable[[InvocationHome, RoleContext], None]
    binaries: tuple[str, ...]


@dataclass(frozen=True)
class HostRuntime:
    """What a host turn needs that the runner image otherwise supplies.

    None of these have container defaults that work here: the analyzer and
    callback URLs name `host.docker.internal`, and both MCP paths are baked
    into the image.
    """

    analyzer_base_url: str
    managed_backend_url: str
    mcp_python: str
    mcp_server: Path
    git: GitRunner
    # Directories outside the worktree that a turn legitimately writes: the
    # cache and temporary roots `uv run` needs, and the profiling environments.
    workspace_roots: tuple[Path, ...] = ()
    # Granting the Docker socket is equivalent to granting root. It is here
    # because 57 of the 81 profiled kernels run through a containerized
    # environment, which is the main reason host execution exists at all.
    unix_sockets: tuple[str, ...] = ("/var/run/docker.sock",)


class RuntimeService:
    def __init__(
        self,
        *,
        workspace: Callable[[str], WorkspaceRuntime],
        providers: ProviderRegistry,
        runtimes: Mapping[str, ProviderRuntime],
        containers: ContainerManager,
        prompts: Prompts,
        mcp: Path,
        container_root: str,
        namespace: str,
        conversation_path: Callable[[str, str], Path] | None = None,
        context_directory: Callable[[str, str], Path] | None = None,
        prepare_workspace: Callable[[str], str] | None = None,
        host: HostRuntime | None = None,
        logger: logging.Logger | None = None,
    ):
        if not namespace:
            raise ValueError("runtime namespace is required")
        self.workspace = workspace
        self.providers = providers
        self.runtimes = dict(runtimes)
        self.containers = containers
        self.prompts = prompts
        self.mcp = mcp
        self.container_root = container_root
        self.namespace = namespace
        self.conversation_path = conversation_path
        self.context_directory = context_directory
        self.prepare_workspace = prepare_workspace
        self.host = host
        self.logger = logger or logging.getLogger("vibesim_agent.runtime")
        self.context_target = (
            managed_context_target(
                containers.environment.managed_context,
                containers.environment.container,
                container_root,
            ).parent
            if context_directory is not None
            else None
        )

    def home(self, request: AgentRequest) -> InvocationHome:
        return self._home(
            request.workspace_id,
            request.conversation_id,
            request.role,
            request.selection.session_scope,
        )

    def _home(self, workspace_id, conversation_id, role, scope) -> InvocationHome:
        return role_home(
            self._root(workspace_id, conversation_id), self.container_root, role, scope
        )

    def _root(self, workspace_id, conversation_id) -> Path:
        validate_conversation_id(conversation_id)
        workspace = self.workspace(workspace_id)
        expected = workspace.state / "runtime" / conversation_id
        root = (
            self.conversation_path(workspace_id, conversation_id)
            if self.conversation_path is not None
            else expected
        )
        if not root.resolve().is_relative_to(workspace.state.resolve()):
            raise ValueError("conversation runtime path escapes workspace state")
        if root.resolve() == workspace.state.resolve():
            raise ValueError("conversation runtime path is the workspace state root")
        if root != expected or root.is_symlink() or expected.parent.is_symlink():
            raise ValueError(
                "conversation runtime path must name its own real directory"
            )
        return root

    def _identity(self, workspace_id, conversation_id):
        owner = json.dumps([self.namespace, workspace_id, conversation_id])
        return "vibesim-agent-" + hashlib.sha256(owner.encode()).hexdigest()[:24], owner

    def _prepare(self, request: TurnInput) -> Execution:
        workspace = self.workspace(request.workspace_id)
        active = []
        # Validate all roles before refreshing profiles or creating a container.
        for role in request.mode.roles:
            runtime = request.runtimes[role]
            selected = self.providers.select(
                runtime.provider_id,
                runtime.model_id,
                effort=runtime.effort,
                service_tier=runtime.service_tier,
            )
            if selected.session_scope != runtime.session_scope:
                raise ValueError("stored role scope is incompatible with provider")
            if not self.providers.available(runtime.provider_id):
                raise ProviderUnavailable((runtime.provider_id,))
            try:
                provision = self.runtimes[runtime.provider_id]
            except KeyError:
                raise ValueError("provider runtime supply is not configured") from None
            home = self._home(
                request.workspace_id,
                request.conversation_id,
                role,
                runtime.session_scope,
            )
            active.append((role, runtime, provision, home))

        directory = (
            self.context_directory(request.workspace_id, request.conversation_id)
            if self.context_directory is not None
            else None
        )
        prompts = self._prompts(workspace)
        prompt = prompts.contract_path(request.mode, request.autonomous)
        if self.prepare_workspace is not None:
            prepared = Path(self.prepare_workspace(request.workspace_id))
            if prepared != workspace.repo:
                raise ValueError("prepared workspace does not match runtime repository")
        # The one place the mode is decided, on the axis the registry already
        # records: a copy of the tracked files runs in a container, a real git
        # tree runs here so that profiling, Slurm and Docker are reachable.
        if workspace.storage_kind == "external":
            return self._host(
                workspace, active, directory=directory, prompt=prompt, prompts=prompts
            )
        return self._container(request, workspace, active, directory, prompt)

    def _host(self, workspace, active, *, directory, prompt, prompts) -> Execution:
        if self.host is None:
            raise ValueError("host execution is not configured")
        check_host_binaries(
            (binary for _, _, provision, _ in active for binary in provision.binaries),
            logger=self.logger,
        )
        environment = self.containers.environment
        permissions = host_permissions(
            # Absolute, and asked of this repository: in a worktree `.git` is a
            # file pointing outside the tree, and the profile has to name the
            # real directories or `git commit` fails on a read-only index.lock.
            git_dir=Path(self._git(workspace.repo, "--absolute-git-dir")),
            git_common_dir=self._common_dir(workspace.repo),
            workspace_roots=self.host.workspace_roots,
            unix_sockets=self.host.unix_sockets,
        )
        context = RoleContext(
            skills=str(workspace.repo / "skills"),
            # The same directory twice here, and two different strings in a
            # container: a profile that links the skills one by one needs to
            # read this side and write the CLI's side.
            skills_source=workspace.repo / "skills",
            # Only the host delivers the contract this way. In a container it
            # arrives as a mount over the workspace's own AGENTS.md instead.
            global_prompt=prompt,
            codex_config=permissions.config,
        )
        for _, _, provision, home in active:
            provision.prepare(home, context)
        managed_context = environment.managed_context
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            managed_context = str(directory / PurePosixPath(managed_context).name)
        return HostExecution(
            repo=workspace.repo,
            agent_prompt=str(prompt),
            schema_directory=str(prompts.directory),
            managed_context=managed_context,
            analyzer_source=environment.agent.analyzer_source,
            analyzer_base_url=self.host.analyzer_base_url,
            managed_backend_url=self.host.managed_backend_url,
            mcp_python=self.host.mcp_python,
            mcp_server=str(self.host.mcp_server),
            permissions=permissions,
        )

    def prompts_for(self, request: TurnInput) -> Prompts:
        """The prompts a turn's roles are given, naming the paths they can see."""
        return self._prompts(self.workspace(request.workspace_id)).bound(
            autonomous=request.autonomous
        )

    def _prompts(self, workspace: WorkspaceRuntime) -> Prompts:
        # A container sees every repository at `/workspace`, so one rendering
        # serves them all. On the host that path is not the repository -- it
        # may not exist, or may be someone else's -- so each workspace gets a
        # rendering that names its own tree, kept with the workspace's state.
        if workspace.storage_kind != "external":
            return self.prompts
        return Prompts.prepare(workspace.state / "prompts", workspace=workspace.repo)

    def _git(self, repo: Path, *arguments: str) -> str:
        # `GitRunner` returns stdout verbatim, and a trailing newline inside a
        # TOML key would produce a profile Codex refuses to load.
        return self.host.git(repo, "rev-parse", *arguments).strip()

    def _common_dir(self, repo: Path) -> Path:
        # `--git-common-dir` is relative to the repository unless it is already
        # absolute, and the profile only accepts absolute paths.
        common = Path(self._git(repo, "--git-common-dir"))
        return common if common.is_absolute() else (repo / common).resolve()

    def _container(self, request, workspace, active, directory, prompt) -> Execution:
        mounts = list(
            workspace_mounts(
                workspace=workspace.repo,
                main=self.containers.environment.agent.main_dir,
                submodules=workspace.submodules,
                prompts=self.prompts.directory,
                agent_prompt=prompt,
                mcp=self.mcp,
                settings=self.containers.environment.container,
                peer_workspace=Path(request.peer_workspace).expanduser()
                if request.peer_workspace
                else None,
            )
        )
        # Derived here rather than from the Execution, which does not exist
        # until the container has been ensured -- after the homes are ready.
        context = RoleContext(
            skills=str(PurePosixPath(WORKSPACE_TARGET) / "skills"),
            skills_source=workspace.repo / "skills",
        )
        for _, _, provision, home in active:
            provision.prepare(home, context)
            mounts.append(Mount(home.host, home.container))
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            mounts.append(Mount(directory, self.context_target, read_only=True))
        name, owner = self._identity(request.workspace_id, request.conversation_id)
        spec = ContainerSpec(
            name=name,
            owner=owner,
            mounts=tuple(mounts),
            role_homes=tuple(home.container for _, _, _, home in active),
            binaries=tuple(
                dict.fromkeys(
                    binary
                    for _, _, provision, _ in active
                    for binary in provision.binaries
                )
            ),
            agent_prompt=str(PROMPTS_TARGET / prompt.name),
            session_scopes=tuple(
                (role.value, runtime.session_scope) for role, runtime, _, _ in active
            ),
        )
        return DockerExecution(
            self.containers.environment, self.containers.ensure(spec)
        )

    def _cleanup(self, workspace_id: str, conversation_id: str) -> None:
        root = self._root(workspace_id, conversation_id)
        if root.exists() and not root.is_dir():
            raise ValueError("conversation runtime cleanup requires a real directory")
        self._release(workspace_id, conversation_id)
        if root.exists():
            shutil.rmtree(root)

    def _release(self, workspace_id: str, conversation_id: str) -> None:
        """Let go of whatever the conversation's last turn was still holding."""
        root = self._root(workspace_id, conversation_id)
        if self.workspace(workspace_id).storage_kind == "external":
            # Deliberately never reaches Docker. A host-only deployment may not
            # have it installed at all, and `ContainerManager` does not catch
            # `FileNotFoundError`, so asking would take the backend down during
            # startup recovery rather than recovering anything.
            if root.exists():
                reap_process_groups(root, logger=self.logger)
            return
        name, owner = self._identity(workspace_id, conversation_id)
        self.containers.remove(name, owner=owner)

    async def prepare(self, request: TurnInput) -> Execution:
        return await self._thread(self._prepare, request)

    async def cleanup(self, workspace_id: str, conversation_id: str) -> None:
        await self._thread(self._cleanup, workspace_id, conversation_id)

    async def release(self, workspace_id: str, conversation_id: str) -> None:
        await self._thread(self._release, workspace_id, conversation_id)

    @staticmethod
    async def _thread(operation, *args):
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Docker and filesystem work cannot be interrupted by cancelling a
            # Python thread. Finish it before releasing the workspace turn lock.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            task.result()
            raise
