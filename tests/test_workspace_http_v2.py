import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from pydantic import SecretStr
from tests import test_application as fixtures
from vibesim_agent.storage.database import Database


class WorkspaceHttpTests(unittest.IsolatedAsyncioTestCase):
    app = fixtures.ApplicationTests.app
    providers = fixtures.ApplicationTests.providers

    def setUp(self):
        fixtures.ApplicationTests.setUp(self)
        self.environment = {
            **{
                key: value
                for key, value in os.environ.items()
                if not key.startswith("GIT_")
            },
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        self.git("init", "--template=")
        self.git("add", "-A")
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "source",
        )
        self.settings = self.settings.model_copy(
            update={
                "agent": self.settings.agent.model_copy(
                    update={"api_token": SecretStr("token")}
                )
            }
        )

    def git(self, *arguments):
        return subprocess.run(
            ["git", "-C", str(self.repo), *arguments],
            env=self.environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    async def asyncSetUp(self):
        Database.create(self.state / "workspace.sqlite")
        self.application = self.app()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.application), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.addAsyncCleanup(self.application.state.turns.close)
        self.base = "/api/agent/v1/workspaces"

    async def create(self, **body):
        response = await self.client.post(
            self.base, json={"displayName": "Trial", **body}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_create_ready_workspace_then_eager_conversation_and_turn(self):
        (self.repo / "AGENTS.md").write_text("dirty instructions")
        (self.repo / "untracked").write_text("do not copy")
        before = self.git("status", "--porcelain")
        descriptor = await self.create(displayName="  Trial  ", autoName=True)
        wid = descriptor["workspace_id"]
        self.assertEqual(descriptor["display_name"], "Trial")
        self.assertEqual(descriptor["naming_state"], "pending")
        self.assertEqual(descriptor["base_revision"], self.git("rev-parse", "HEAD"))
        registry = self.application.state.workspaces
        repository = registry.repo_path(wid)
        self.assertEqual((repository / "AGENTS.md").read_text(), "dirty instructions")
        self.assertFalse((repository / "untracked").exists())
        with Database(registry.database_path(wid)).connect() as connection:
            self.assertEqual(
                connection.execute("SELECT count(*) FROM conversations").fetchone()[0],
                0,
            )
        conversation = await self.client.post(
            f"{self.base}/{wid}/conversations",
            json={"agentMode": "single", "eager": True},
        )
        self.assertEqual(conversation.status_code, 200, conversation.text)
        self.assertEqual(conversation.json()["workspace_path"], str(repository))
        path = f"{self.base}/{wid}/conversations/{conversation.json()['id']}"
        response = await self.client.post(
            path + "/messages", json={"text": "hello", "agentMode": "single"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len((await self.client.get(path)).json()["messages"]), 2)
        self.assertEqual(self.git("status", "--porcelain"), before)
        self.assertGreaterEqual(
            registry.get(wid)["last_accessed_at"], descriptor["last_accessed_at"]
        )
        index = json.loads(registry.registry_path.read_text())
        row = next(row for row in index["workspaces"] if row["workspace_id"] == wid)
        self.assertEqual(row["logs_root"], f"{wid}/repo/logs")

    async def test_rename_archive_and_manual_name_wins(self):
        descriptor = await self.create(autoName=True)
        wid = descriptor["workspace_id"]
        path = f"{self.base}/{wid}"
        changed = await self.client.patch(
            path, json={"displayName": " Chosen ", "state": "archived"}
        )
        self.assertEqual(changed.status_code, 200, changed.text)
        self.assertEqual(changed.json()["display_name"], "Chosen")
        self.assertEqual(changed.json()["naming_state"], "manual")
        self.assertFalse(
            self.application.state.workspaces.apply_generated_name(wid, "Late")
        )
        self.assertEqual((await self.client.get(path)).status_code, 200)
        rows = (await self.client.get(self.base)).json()["workspaces"]
        self.assertNotIn(wid, [row["workspace_id"] for row in rows])
        for body in ({"displayName": " "}, {"state": "invalid"}):
            self.assertEqual(
                (await self.client.patch(path, json=body)).status_code, 400
            )
        self.assertEqual(
            (
                await self.client.patch(
                    self.base + "/w_main", json={"state": "archived"}
                )
            ).status_code,
            400,
        )
        self.assertEqual(
            (
                await self.client.patch(
                    self.base + "/w_missing", json={"displayName": "x"}
                )
            ).status_code,
            404,
        )

    async def test_first_message_prepares_an_existing_descriptor_with_missing_repo(self):
        descriptor = await self.create()
        wid = descriptor["workspace_id"]
        repository = self.application.state.workspaces.repo_path(wid)
        shutil.rmtree(repository)
        response = await self.client.post(
            f"{self.base}/{wid}/conversations", json={"agentMode": "single"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(repository.exists())
        path = f"{self.base}/{wid}/conversations/{response.json()['id']}"
        sent = await self.client.post(path + "/messages", json={"text": "hello"})
        self.assertEqual(sent.status_code, 200, sent.text)
        history = (await self.client.get(path)).json()
        self.assertEqual(len(history["messages"]), 2)
        self.assertEqual(history["messages"][-1]["content"], "Done")
        self.assertTrue((repository / ".git").is_dir())

    async def test_tools_auth_and_workspace_input_contract(self):
        tools = "/api/agent/v1/tools/workspaces"
        for method in (self.client.get, self.client.post):
            self.assertEqual((await method(tools)).status_code, 401)
        headers = {"Authorization": "Bearer token"}
        self.assertEqual(
            (await self.client.post(tools, headers=headers, json={})).status_code, 422
        )
        response = await self.client.post(
            tools, headers=headers, json={"display_name": "Tool", "auto_name": True}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["naming_state"], "pending")
        listed = (await self.client.get(tools, headers=headers)).json()
        self.assertEqual(listed, (await self.client.get(self.base)).json())
        self.assertEqual(
            (await self.client.post(self.base, json={"displayName": " "})).status_code,
            400,
        )

    async def test_failed_preparation_or_database_creation_leaves_no_workspace(self):
        service = self.application.state.workspace_service
        before = set(service.registry.root.iterdir())
        for target in ("snapshot", "database"):
            with self.subTest(target=target):
                failure = (
                    patch.object(
                        service.snapshot, "populate", side_effect=OSError("copy failed")
                    )
                    if target == "snapshot"
                    else patch(
                        "vibesim_agent.services.workspace.Database.create",
                        side_effect=OSError("database failed"),
                    )
                )
                with failure, self.assertRaises(OSError):
                    service.create("Failed", workspace_id="w_failed")
                self.assertEqual(set(service.registry.root.iterdir()), before)
                self.assertEqual(
                    [row["workspace_id"] for row in service.registry.list()], ["w_main"]
                )

    async def test_prepare_reuses_managed_repo_and_repairs_unprepared_descriptor(self):
        descriptor = await self.create()
        wid = descriptor["workspace_id"]
        service = self.application.state.workspace_service
        repository = Path(service.prepare(wid))
        (repository / "local").write_text("keep local edits")
        self.assertEqual(service.prepare(wid), str(repository))
        self.assertEqual((repository / "local").read_text(), "keep local edits")
        before = self.git("status", "--porcelain")
        self.assertEqual(service.prepare("w_main"), str(self.repo))
        self.assertEqual(self.git("status", "--porcelain"), before)
        shutil.rmtree(repository)
        self.assertEqual(service.prepare(wid), str(repository))
        self.assertTrue((repository / ".git").is_dir())
        with Database(service.registry.database_path(wid)).connect():
            pass
        shutil.rmtree(repository)
        repository.symlink_to(self.repo, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "managed repository"):
            service.prepare(wid)
