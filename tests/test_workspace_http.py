import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from pydantic import SecretStr
from tests import test_application as fixtures
from vibesim_agent.api.workspaces import workspace_collection_router
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


class WorktreeWorkspaceHttpTests(WorkspaceHttpTests):
    """Worktree creation over HTTP, against the fixture's real Git checkout."""

    def setUp(self):
        super().setUp()
        self.trees = self.root / "trees"
        self.trees.mkdir()
        self.settings = self.settings.model_copy(
            update={
                "agent": self.settings.agent.model_copy(
                    update={"worktree_root": self.trees}
                )
            }
        )

    async def create_worktree(self, **body):
        return await self.client.post(
            self.base, json={"displayName": "Kv Cache Logging", "kind": "worktree", **body}
        )

    async def test_capability_is_announced_beside_the_listing(self):
        listing = (await self.client.get(self.base)).json()
        self.assertEqual(
            listing["capabilities"]["workspaceKinds"], ["copy", "worktree"]
        )

    async def test_creates_a_real_branch_and_registers_it_external(self):
        response = await self.create_worktree()
        self.assertEqual(response.status_code, 200, response.text)
        descriptor = response.json()

        self.assertEqual(descriptor["storage_kind"], "external")
        self.assertEqual(descriptor["workspace_kind"], "worktree")
        self.assertIs(descriptor["worktree_owned"], True)
        # Derived from the display name, and the server reports what it used;
        # the UI must never re-derive this.
        self.assertEqual(descriptor["worktree_branch"], "kv-cache-logging")
        self.assertEqual(descriptor["repo_path"], str(self.trees / "wt-kv-cache-logging"))
        self.assertIn("kv-cache-logging", self.git("branch", "--list"))
        self.assertEqual(descriptor["base_revision"], self.git("rev-parse", "HEAD"))
        self.assertEqual(
            (Path(descriptor["repo_path"]) / "AGENTS.md").read_text(),
            "workspace instructions",
        )

    async def test_second_workspace_with_the_same_name_gets_its_own_branch(self):
        first = (await self.create_worktree()).json()
        second = (await self.create_worktree()).json()
        self.assertEqual(first["worktree_branch"], "kv-cache-logging")
        self.assertEqual(second["worktree_branch"], "kv-cache-logging-2")
        self.assertNotEqual(first["repo_path"], second["repo_path"])

    async def test_explicit_branch_collision_is_a_conflict_not_a_rename(self):
        await self.create_worktree(branch="chosen")
        response = await self.create_worktree(branch="chosen")
        self.assertEqual(response.status_code, 409, response.text)

    async def test_invalid_branch_is_rejected_before_anything_is_created(self):
        response = await self.create_worktree(branch="bad branch")
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.git("worktree", "list").count("\n"), 0)

    async def test_branch_and_base_do_not_apply_to_a_copy(self):
        response = await self.client.post(
            self.base, json={"displayName": "Trial", "kind": "copy", "branch": "x"}
        )
        self.assertEqual(response.status_code, 400, response.text)

    async def test_unknown_kind_is_refused_by_the_schema(self):
        response = await self.client.post(
            self.base, json={"displayName": "Trial", "kind": "adopt"}
        )
        self.assertEqual(response.status_code, 422, response.text)


    # The naming model proposes the branch before the worktree exists. These
    # mount the router alone, over the same service and real Git, with a
    # stand-in for the model.
    def naming(self, proposal):
        self.proposals = []
        self.proposal = proposal

        async def propose(request):
            self.proposals.append(request)
            if isinstance(self.proposal, Exception):
                raise self.proposal
            return self.proposal

        named = FastAPI()
        named.include_router(
            workspace_collection_router(
                self.application.state.workspace_service, branch_topic=propose
            )
        )
        self.named = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=named), base_url="http://test"
        )
        self.addAsyncCleanup(self.named.aclose)

    async def named_worktree(self, **body):
        return await self.named.post(
            self.base, json={"displayName": "Kv Cache Logging", "kind": "worktree", **body}
        )

    async def test_the_models_branch_is_used_in_place_of_the_display_name(self):
        self.naming("kv-cache-eviction-logging")
        descriptor = (await self.named_worktree()).json()
        self.assertEqual(self.proposals, ["Kv Cache Logging"])
        self.assertEqual(descriptor["worktree_branch"], "kv-cache-eviction-logging")
        self.assertEqual(
            descriptor["repo_path"], str(self.trees / "wt-kv-cache-eviction-logging")
        )
        # The display name is still the reader's, untouched by the branch.
        self.assertEqual(descriptor["display_name"], "Kv Cache Logging")

    async def test_a_proposal_is_slugged_and_suffixed_rather_than_refused(self):
        self.naming("KV Cache: Eviction Logging!")
        first = (await self.named_worktree()).json()
        second = (await self.named_worktree()).json()
        # Unlike a typed branch, a collision here is nobody's mistake.
        self.assertEqual(first["worktree_branch"], "kv-cache-eviction-logging")
        self.assertEqual(second["worktree_branch"], "kv-cache-eviction-logging-2")

    async def test_a_failed_or_empty_proposal_falls_back_to_the_display_name(self):
        self.naming(None)
        for proposal in (TimeoutError("slow"), "", "！？"):
            with self.subTest(proposal=proposal):
                self.proposal = proposal
                response = await self.named_worktree()
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(
                    response.json()["worktree_branch"].startswith("kv-cache-logging")
                )

    async def test_a_typed_branch_is_never_replaced_or_even_asked_about(self):
        self.naming("unused")
        descriptor = (await self.named_worktree(branch="chosen")).json()
        self.assertEqual(descriptor["worktree_branch"], "chosen")
        self.assertEqual(self.proposals, [])

    async def test_a_copy_does_not_ask_for_a_branch(self):
        self.naming("unused")
        await self.named.post(self.base, json={"displayName": "Trial", "kind": "copy"})
        self.assertEqual(self.proposals, [])


class WorktreeDisabledHttpTests(WorkspaceHttpTests):
    """Unset `worktree_root` is the default; the feature must stay off and say so."""

    async def test_capability_omits_worktree_and_creation_is_not_implemented(self):
        listing = (await self.client.get(self.base)).json()
        self.assertEqual(listing["capabilities"]["workspaceKinds"], ["copy"])
        response = await self.client.post(
            self.base, json={"displayName": "Trial", "kind": "worktree"}
        )
        self.assertEqual(response.status_code, 501, response.text)

    async def test_copies_still_record_their_kind(self):
        descriptor = await self.create()
        self.assertEqual(descriptor["workspace_kind"], "copy")
        self.assertEqual(descriptor["storage_kind"], "managed")

    async def test_main_checkout_reports_a_kind_it_never_stored(self):
        listing = (await self.client.get(self.base)).json()["workspaces"]
        main = next(item for item in listing if item["workspace_id"] == "w_main")
        # w_main's descriptor predates the kind axis entirely.
        self.assertEqual(main["workspace_kind"], "checkout")
        # Derived from storage_kind, which is why a descriptor written before
        # host execution existed still reports the mode it now runs in.
        self.assertEqual(main["execution"], "host")
        self.assertNotIn("workspace_kind", self.application.state.workspaces.get("w_main"))


class SlugTests(unittest.TestCase):
    def test_cuts_at_a_word_boundary(self):
        from vibesim_agent.services.workspace import _slug

        slug = _slug("When serving GLM5.2 with TP4 + EP4, which kernel takes the most time")
        self.assertEqual(slug, "when-serving-glm5-2-with-tp4-ep4-which")
        self.assertLessEqual(len(slug), 40)
        self.assertEqual(_slug("a" * 60), "a" * 40)
        self.assertEqual(_slug("为什么这么慢"), "workspace")
