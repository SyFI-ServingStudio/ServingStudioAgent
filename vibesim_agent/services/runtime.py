"""Prepare declared provider homes and one owned container for a turn."""

import asyncio
import hashlib
import json
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
from ..runtime.execution import DockerExecution, Execution
from ..runtime.homes import role_home
from ..runtime.invocation import InvocationHome, RoleContext
from ..runtime.mounts import (
    PROMPTS_TARGET,
    WORKSPACE_TARGET,
    Mount,
    managed_context_target,
    workspace_mounts,
)


@dataclass(frozen=True)
class WorkspaceRuntime:
    repo: Path
    state: Path
    submodules: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ProviderRuntime:
    prepare: Callable[[InvocationHome, RoleContext], None]
    binaries: tuple[str, ...]


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
        prompt = self.prompts.contract_path(request.mode, request.autonomous)
        if self.prepare_workspace is not None:
            prepared = Path(self.prepare_workspace(request.workspace_id))
            if prepared != workspace.repo:
                raise ValueError("prepared workspace does not match runtime repository")
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
        context = RoleContext(skills=str(PurePosixPath(WORKSPACE_TARGET) / "skills"))
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
        # The only place the mode is decided. Host execution has not been
        # switched on yet, so every turn is still a container.
        return DockerExecution(self.containers.environment, self.containers.ensure(spec))

    def _cleanup(self, workspace_id: str, conversation_id: str) -> None:
        root = self._root(workspace_id, conversation_id)
        if root.exists() and not root.is_dir():
            raise ValueError("conversation runtime cleanup requires a real directory")
        self._remove_container(workspace_id, conversation_id)
        if root.exists():
            shutil.rmtree(root)

    def _remove_container(self, workspace_id: str, conversation_id: str) -> None:
        self._root(workspace_id, conversation_id)
        name, owner = self._identity(workspace_id, conversation_id)
        self.containers.remove(name, owner=owner)

    async def prepare(self, request: TurnInput) -> Execution:
        return await self._thread(self._prepare, request)

    async def cleanup(self, workspace_id: str, conversation_id: str) -> None:
        await self._thread(self._cleanup, workspace_id, conversation_id)

    async def remove_container(self, workspace_id: str, conversation_id: str) -> None:
        await self._thread(self._remove_container, workspace_id, conversation_id)

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
