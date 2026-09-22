import asyncio
import os
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from tests.runtime_fixtures import execution_environment
from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import AgentMode, Role
from vibesim_agent.domain.turns import TurnInput
from vibesim_agent.prompts.render import Prompts
from vibesim_agent.providers.base import AgentRequest, Model, Provider
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.runtime.git import GitRunner
from vibesim_agent.runtime.host import HostUnavailable
from vibesim_agent.services.runtime import (
    HostRuntime,
    ProviderRuntime,
    RuntimeService,
    WorkspaceRuntime,
)
from tests.test_application import git_init
from vibesim_agent.settings import ProviderSettings


class RuntimeServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "AGENTS.md").write_text("tracked instructions")
        self.mcp = self.root / "mcp"
        self.mcp.mkdir()
        self.prompts = Prompts.prepare(self.root / "prompts")
        environment = execution_environment()
        environment = replace(
            environment,
            agent=environment.agent.model_copy(update={"main_dir": self.repo}),
        )
        self.containers = Mock(environment=environment)
        self.containers.ensure.return_value = "container-id"
        self.providers = ProviderRegistry()
        self.prepared = []

        def prepare(home, context):
            home.host.mkdir(parents=True, exist_ok=True)
            self.prepared.append(home)

        self.runtimes = {}
        selections = {}
        for role in Role:
            provider_id = role.value
            model = Model(provider_id, provider_id, ("high",), "high")
            self.providers.register(
                Provider(
                    provider_id,
                    provider_id,
                    Mock(adapter_id="test"),
                    ProviderSettings(model=provider_id, effort="high"),
                    provider_id + ":scope",
                    lambda model=model: (model,),
                )
            )
            self.runtimes[provider_id] = ProviderRuntime(prepare, ("test-cli",))
            selections[role] = RoleRuntime(
                provider_id, provider_id + ":scope", provider_id, "high", "default"
            )
        self.runtime = RuntimeService(
            workspace=lambda workspace_id: WorkspaceRuntime(
                self.repo, self.root / workspace_id
            ),
            providers=self.providers,
            runtimes=self.runtimes,
            containers=self.containers,
            prompts=self.prompts,
            mcp=self.mcp,
            container_root="/runtime",
            namespace="test-namespace",
        )
        self.request = TurnInput(
            "workspace",
            "conversation",
            "turn",
            "question",
            AgentMode.ORCHESTRATED,
            selections,
            {},
            "",
        )

    def test_prepares_active_roles_and_passes_complete_mounts_and_scope(self):
        execution = self.runtime._prepare(self.request)
        self.assertEqual(execution.container, "container-id")
        # Host execution is not switched on yet; every turn is still a container.
        self.assertIsNone(execution.cwd)
        spec = self.containers.ensure.call_args.args[0]
        self.assertEqual(len(self.prepared), 2)
        self.assertEqual(spec.binaries, ("test-cli",))
        self.assertEqual(
            spec.session_scopes,
            (
                ("orchestrator", "orchestrator:scope"),
                ("implementer", "implementer:scope"),
            ),
        )
        self.assertEqual(spec.agent_prompt, "/opt/vibesim/prompts/AGENTS.md")
        for role, home in zip(self.request.mode.roles, self.prepared, strict=True):
            selection = self.providers.select(role.value)
            call = AgentRequest(
                "workspace",
                "conversation",
                "turn",
                role,
                "question",
                "container-id",
                selection,
            )
            self.assertEqual(self.runtime.home(call), home)
            mount = next(m for m in spec.mounts if str(m.target) == home.container)
            self.assertEqual(mount.source, home.host)
            self.assertFalse(mount.read_only)
        self.assertEqual((self.repo / "AGENTS.md").read_text(), "tracked instructions")

    def test_single_only_supplies_assistant_and_reuses_identity_across_turns(self):
        request = replace(self.request, mode=AgentMode.SINGLE)
        self.runtime._prepare(request)
        one = self.containers.ensure.call_args.args[0]
        self.runtime._prepare(replace(request, turn_id="next"))
        two = self.containers.ensure.call_args.args[0]
        self.assertEqual(one, two)
        self.assertEqual(one.session_scopes, (("assistant", "assistant:scope"),))
        self.runtime._prepare(replace(request, conversation_id="other"))
        other = self.containers.ensure.call_args.args[0]
        self.assertNotEqual(one.name, other.name)
        self.assertNotEqual(one.owner, other.owner)

    def test_invalid_scope_and_missing_supply_fail_before_any_profile_write(self):
        runtimes = dict(self.request.runtimes)
        runtimes[Role.IMPLEMENTER] = replace(
            runtimes[Role.IMPLEMENTER], session_scope="bad"
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.runtime._prepare(replace(self.request, runtimes=runtimes))
        self.runtime.runtimes.pop("implementer")
        with self.assertRaisesRegex(ValueError, "not configured"):
            self.runtime._prepare(self.request)
        self.assertEqual(self.prepared, [])
        self.containers.ensure.assert_not_called()

    def test_invalid_conversation_path_cannot_escape_state(self):
        for identity in ("../other", "", "..", "a/b", "a\\b", "nul\x00", "a" * 81):
            with (
                self.subTest(identity=identity),
                self.assertRaisesRegex(ValueError, "conversation ID"),
            ):
                self.runtime._prepare(replace(self.request, conversation_id=identity))
        self.assertEqual(self.prepared, [])

    def test_legacy_conversation_identity_remains_usable(self):
        for identity in ("old:conversation", "old conversation"):
            with self.subTest(identity=identity):
                self.runtime._prepare(replace(self.request, conversation_id=identity))
                self.assertIn(identity, self.containers.ensure.call_args.args[0].owner)
                self.assertTrue(
                    all(
                        home.host.parent.parent.name == identity
                        for home in self.prepared[-2:]
                    )
                )

    async def test_cancel_waits_for_thread_completion_even_when_cancelled_again(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def ensure(spec):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test preparation was not released")
            return "container-id"

        self.containers.ensure.side_effect = ensure
        task = asyncio.create_task(self.runtime.prepare(self.request))
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        self.assertFalse(task.done())
        task.cancel()
        await asyncio.sleep(0.01)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertEqual(self.containers.ensure.call_count, 1)

    async def test_preparation_failure_after_stop_is_not_hidden_as_cancellation(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def ensure(spec):
            entered.set()
            release.wait(3)
            raise RuntimeError("container cleanup failed")

        self.containers.ensure.side_effect = ensure
        task = asyncio.create_task(self.runtime.prepare(self.request))
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        release.set()
        with self.assertRaisesRegex(RuntimeError, "container cleanup failed"):
            await asyncio.wait_for(task, 2)


class ExecutionModeTests(unittest.TestCase):
    """The one branch: a copy of the tracked files, or a real git tree."""

    def setUp(self):
        RuntimeServiceTests.setUp(self)
        # A real repository, because the host branch asks git where this tree's
        # `.git` is. Without the `init` the answer came from whichever ancestor
        # of the temporary directory happened to be a checkout -- which passed
        # only while TMPDIR sat inside this workspace.
        git_init(self.repo)

    def service(self, storage_kind, **overrides):
        state = self.root / "state"
        context = state / "runtime" / "conversation" / "managed"
        arguments = {
            "workspace": lambda workspace_id: WorkspaceRuntime(
                self.repo, state, storage_kind=storage_kind
            ),
            "providers": self.providers,
            "runtimes": self.runtimes,
            "containers": self.containers,
            "prompts": self.prompts,
            "mcp": self.mcp,
            "container_root": "/runtime",
            "namespace": "test-namespace",
            "context_directory": lambda workspace_id, conversation_id: context,
            "host": HostRuntime(
                analyzer_base_url="http://172.17.0.1:63044",
                managed_backend_url="http://172.17.0.1:63043",
                mcp_python="/agent/.venv/bin/python",
                mcp_server=self.root / "server.py",
                git=GitRunner({"PATH": os.environ.get("PATH", "")}),
                workspace_roots=(self.root,),
            ),
            **overrides,
        }
        self.context = context
        return RuntimeService(**arguments)

    def prepare(self, storage_kind, **overrides):
        return self.service(storage_kind, **overrides)._prepare(
            replace(self.request, conversation_id="conversation")
        )

    def test_a_managed_copy_mounts_the_context_directory_read_only(self):
        execution = self.prepare("managed")
        self.assertEqual(execution.container, "container-id")
        self.assertIsNone(execution.cwd)
        spec = self.containers.ensure.call_args.args[0]
        [mount] = [m for m in spec.mounts if m.source == self.context]
        # Replaceable capability files: the runner must not be able to forge one.
        self.assertTrue(mount.read_only)
        self.assertEqual(str(mount.target), "/opt/vibesim/managed")
        self.assertEqual(execution.managed_context, "/opt/vibesim/managed/context.json")

    def test_a_real_tree_runs_here_with_the_context_as_a_host_path(self):
        with patch("vibesim_agent.services.runtime.check_host_binaries") as check:
            execution = self.prepare("external")
        self.containers.ensure.assert_not_called()
        self.assertEqual(execution.cwd, self.repo)
        self.assertEqual(list(check.call_args.args[0]), ["test-cli", "test-cli"])
        # The container constant names a mount target that does not exist here.
        self.assertEqual(
            execution.managed_context, str(self.context / "context.json")
        )
        self.assertTrue(self.context.is_dir())
        self.assertEqual(execution.analyzer_base_url, "http://172.17.0.1:63044")
        self.assertEqual(execution.managed_backend_url, "http://172.17.0.1:63043")
        self.assertEqual(execution.mcp_server, str(self.root / "server.py"))
        # Where the role schemas are, from the CLI's side. Codex is handed this
        # path verbatim as `--output-schema`, and the container's mount target
        # is a file that does not exist out here: a host turn built with it
        # exits before it reads its prompt.
        self.assertEqual(
            Path(execution.schema_directory), Path(execution.agent_prompt).parent
        )
        self.assertTrue(
            (Path(execution.schema_directory) / "assistant.schema.json").is_file()
        )

    def test_a_host_turn_is_told_its_own_tree_rather_than_the_mount(self):
        with patch("vibesim_agent.services.runtime.check_host_binaries"):
            execution = self.prepare("external")
        # `/workspace` is a mount target; on a host it is some other directory
        # or none, and an agent told to work there works in the wrong tree.
        contract = Path(execution.agent_prompt).read_text()
        self.assertNotIn("/workspace", contract)
        self.assertIn(f"`{self.repo}/skills/skill-of-skills/SKILL.md`", contract)
        prompts = self.service("external").prompts_for(
            replace(self.request, autonomous=True)
        )
        self.assertIn(
            f"Read and follow `{Path(execution.agent_prompt).with_name('AGENTS.single.autonomous.md')}`",
            prompts.role_text(Role.ASSISTANT),
        )
        self.assertIn(
            f"Conversation plan: `{self.repo}/c_plan.md`.",
            prompts.driver_contract(AgentMode.SINGLE, "c"),
        )

    def test_the_host_contract_reaches_the_role_homes_and_the_repo_skills(self):
        contexts = []
        self.runtimes = {
            name: replace(
                runtime, prepare=lambda home, context: contexts.append(context)
            )
            for name, runtime in self.runtimes.items()
        }
        with patch("vibesim_agent.services.runtime.check_host_binaries"):
            execution = self.prepare("external")
        for context in contexts:
            # Codex reads $CODEX_HOME/AGENTS.md; in a container the same file
            # arrives as a mount instead, so this is host-only.
            self.assertEqual(context.global_prompt, Path(execution.agent_prompt))
            self.assertEqual(context.skills, str(self.repo / "skills"))
            # The same directory read from this side, so a profile can list it.
            self.assertEqual(context.skills_source, self.repo / "skills")
            # The user's own skills are offered here and nowhere else.
            self.assertTrue(context.user_skills)
        self.assertTrue(Path(execution.agent_prompt).is_file())

    def test_a_container_names_the_mount_but_still_says_where_to_read_it(self):
        contexts = []
        def record(home, context):
            # The home still has to exist: it is mounted right after this.
            home.host.mkdir(parents=True, exist_ok=True)
            contexts.append(context)

        self.runtimes = {
            name: replace(runtime, prepare=record)
            for name, runtime in self.runtimes.items()
        }
        self.prepare("managed")
        for context in contexts:
            # Two different strings for one directory: the CLI sees the mount
            # target, and whoever enumerates the skills is out here.
            self.assertEqual(context.skills, "/workspace/skills")
            self.assertEqual(context.skills_source, self.repo / "skills")
            self.assertFalse(context.user_skills)

    def test_a_missing_cli_refuses_the_turn_rather_than_spawning(self):
        with self.assertRaisesRegex(HostUnavailable, "test-cli"):
            self.prepare("external")
        self.assertEqual(self.prepared, [])

    def test_a_deployment_without_host_configuration_refuses_external_turns(self):
        with self.assertRaisesRegex(ValueError, "not configured"):
            self.prepare("external", host=None)
        self.assertEqual(self.prepared, [])


if __name__ == "__main__":
    unittest.main()
