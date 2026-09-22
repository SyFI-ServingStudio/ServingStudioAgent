import json
import os
import subprocess
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from tests import test_runtime_container as docker_fixture
from tests.runtime_fixtures import execution_environment
from vibesim_agent.application import ProviderSetup, build_application
from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import Role
from vibesim_agent.providers.base import Model, Provider
from vibesim_agent.providers.claude.home import ClaudeProfile
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.services.runtime import ProviderRuntime
from vibesim_agent.settings import ProviderSettings, Settings
from vibesim_agent.storage.database import Database, SchemaMismatch


def git_init(repo):
    subprocess.run(
        ["git", "-C", str(repo), "init", "--template=", "-q"],
        env={
            **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        },
        check=True,
        capture_output=True,
        timeout=30,
    )


class ApplicationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.state = self.root / "workspaces/w_main"
        self.state.mkdir(parents=True)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "AGENTS.md").write_text("workspace instructions")
        (self.repo / "uv.lock").write_text("lock contents")
        # `w_main` is external, so its turns run on the host and the Codex
        # permission profile is built from this repository's own git
        # directories. A plain directory would have none.
        git_init(self.repo)
        self.mcp = self.root / "mcp"
        self.mcp.mkdir()
        (self.state / "workspace.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "workspace_id": "w_main",
                    "display_name": "Main",
                    "state": "active",
                    "storage_kind": "external",
                    "repo_path": str(self.repo),
                    "logs_path": str(self.repo / "logs"),
                    "created_at": 1,
                    "last_accessed_at": 1,
                }
            )
        )
        env = execution_environment()
        self.settings = Settings(
            agent=env.agent.model_copy(
                update={"main_dir": self.repo, "workspaces_root": self.state.parent}
            ),
            container=env.container,
            providers={},
        )
        self.docker = docker_fixture.Docker()
        self.calls = []
        self.before = []
        self.factories = 0
        self.docker_environments = []

    def providers(self, home, environment, prompts):
        self.factories += 1
        calls = self.calls

        class Adapter:
            adapter_id = "test"

            async def run(self, request):
                calls.append((request, home(request)))
                yield {
                    "kind": "session",
                    "role": request.role.value,
                    "session_id": "saved-session",
                }
                yield {"kind": "role_ready", "role": request.role.value}
                yield {
                    "kind": "final",
                    "text": '{"action":"final_answer","message":"Done"}',
                }

        model = Model("test-model", "Test", ("high",), "high")
        registry = ProviderRegistry()
        registry.register(
            Provider(
                "test",
                "Test",
                Adapter(),
                ProviderSettings(model="test-model", effort="high"),
                "test:scope",
                lambda: (model,),
            )
        )
        defaults = {
            role: RoleRuntime("test", "test:scope", "test-model", "high", "default")
            for role in Role
        }
        return ProviderSetup(
            registry,
            # A tool that really is on PATH: `w_main` is external, so its turns
            # run on the host and the readiness check refuses a missing one.
            {"test": ProviderRuntime(ClaudeProfile().prepare, ("sh",))},
            defaults,
            {"legacy-model": "test-model"},
            {"DOCKER_HOST": "unix:///explicit.sock"},
        )

    def app(self):
        async def before_call(request):
            self.before.append(request)

        def run(command, **kwargs):
            self.docker_environments.append(kwargs["env"])
            return self.docker(command, **kwargs)

        app = build_application(
            self.settings,
            providers=self.providers,
            before_call=before_call,
            prompts_directory=self.root / "prompts",
            mcp_directory=self.mcp,
            managed_context="/runtime/context.json",
            namespace="test",
            submodules=(),
            run=run,
        )
        self.addAsyncCleanup(app.state.turns.close)
        return app

    async def test_create_send_reload_and_resume_through_assembled_services(self):
        Database.create(self.state / "workspace.sqlite")
        app = self.app()
        self.assertEqual(self.docker.calls, [])
        self.assertFalse((self.state / "runtime").exists())
        base = "/api/agent/v1/workspaces/w_main/conversations"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            created = await client.post(
                base,
                json={
                    "agentMode": "single",
                    "codex_runtime": {"assistant": {"model": "legacy-model"}},
                },
            )
            self.assertEqual(created.status_code, 200, created.text)
            self.assertEqual(
                created.json()["codex_runtime"]["assistant"]["model"], "test-model"
            )
            path = base + "/" + created.json()["id"]
            response = await client.post(path + "/messages", json={"text": "first"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("event: done", response.text)
            history = (await client.get(path)).json()
            self.assertEqual(
                [m["content"] for m in history["messages"]], ["first", "Done"]
            )
        await app.state.turns.close()
        restarted = self.app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restarted), base_url="http://test"
        ) as client:
            response = await client.post(path + "/messages", json={"text": "second"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("event: done", response.text)
            self.assertEqual(len((await client.get(path)).json()["messages"]), 4)
        self.assertEqual(
            [call.session_id for call, _ in self.calls], [None, "saved-session"]
        )
        self.assertEqual(self.calls[0][1], self.calls[1][1])
        self.assertEqual(len(self.before), 2)
        self.assertEqual(
            self.calls[0][1], restarted.state.runtime.home(self.calls[0][0])
        )
        self.assertTrue(self.calls[0][1].host.is_dir())
        # `w_main` is external, so both turns ran here. Nothing asked Docker for
        # anything, which is also what makes this work on a host without it.
        self.assertEqual(self.docker.calls, [])
        self.assertEqual(self.calls[0][0].execution.cwd, self.repo)
        self.assertEqual(
            (self.repo / "AGENTS.md").read_text(), "workspace instructions"
        )

    def managed_workspace(self):
        """A copy beside the external one, published the way the registry would.

        Not in `setUp`: four other test modules borrow it, and a second listed
        workspace changes what they count.
        """
        state = self.root / "workspaces/w_copy"
        (state / "repo").mkdir(parents=True)
        (state / "repo/AGENTS.md").write_text("copied instructions")
        (state / "workspace.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "workspace_id": "w_copy",
                    "display_name": "Copy",
                    "state": "active",
                    "storage_kind": "managed",
                    "repo_path": "repo",
                    "logs_path": "repo/logs",
                    "created_at": 1,
                    "last_accessed_at": 1,
                }
            )
        )
        Database.create(state / "workspace.sqlite")

    async def test_a_managed_workspace_still_assembles_and_runs_a_container(self):
        Database.create(self.state / "workspace.sqlite")
        self.managed_workspace()
        app = self.app()
        base = "/api/agent/v1/workspaces/w_copy/conversations"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            created = await client.post(base, json={"agentMode": "single"})
            self.assertEqual(created.status_code, 200, created.text)
            path = base + "/" + created.json()["id"]
            response = await client.post(path + "/messages", json={"text": "first"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("event: done", response.text)
        self.assertTrue(any(command[1] == "create" for command in self.docker.calls))
        self.assertTrue(self.docker_environments)
        self.assertTrue(
            all(
                env == {"DOCKER_HOST": "unix:///explicit.sock"}
                for env in self.docker_environments
            )
        )
        [(request, _)] = self.calls
        self.assertIsNone(request.execution.cwd)
        self.assertEqual(request.execution.agent_prompt, "/workspace/AGENTS.md")

    def test_old_database_rejected_without_migration_or_provider_start(self):
        path = self.state / "workspace.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE legacy (value TEXT)")
            connection.execute("INSERT INTO legacy VALUES ('preserved')")
        before = path.read_bytes()
        with self.assertRaises(SchemaMismatch):
            self.app()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.factories, 0)
        self.assertEqual(self.docker.calls, [])
        self.assertFalse((self.root / "prompts").exists())

    def test_missing_database_is_not_created(self):
        with self.assertRaises(sqlite3.OperationalError):
            self.app()
        self.assertFalse((self.state / "workspace.sqlite").exists())
        self.assertEqual(self.docker.calls, [])

    async def test_runtime_symlink_escape_fails_before_profile_or_docker_writes(self):
        Database.create(self.state / "workspace.sqlite")
        outside = self.root / "outside"
        outside.mkdir()
        (self.state / "runtime").symlink_to(outside, target_is_directory=True)
        app = self.app()
        base = "/api/agent/v1/workspaces/w_main/conversations"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            created = (await client.post(base, json={"agentMode": "single"})).json()
            path = base + "/" + created["id"]
            response = await client.post(path + "/messages", json={"text": "question"})
            self.assertEqual(response.status_code, 200)
            history = (await client.get(path)).json()
            self.assertEqual(history["messages"][-1]["outcome"], "failed")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(self.docker.calls, [])
        self.assertEqual(self.calls, [])
