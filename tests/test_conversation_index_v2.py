import unittest
from unittest.mock import patch

from tests import test_workspace_http_v2 as fixtures
from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import Role, Sandbox


class ConversationIndexTests(unittest.IsolatedAsyncioTestCase):
    app = fixtures.WorkspaceHttpTests.app
    providers = fixtures.WorkspaceHttpTests.providers
    git = fixtures.WorkspaceHttpTests.git
    setUp = fixtures.WorkspaceHttpTests.setUp
    asyncSetUp = fixtures.WorkspaceHttpTests.asyncSetUp

    async def test_active_index_preserves_workspace_identity_and_stable_order(self):
        service = self.application.state.workspace_service
        for wid in ("w_a", "w_b", "w_archived"):
            service.create(wid, workspace_id=wid)
        runtimes = {
            role: RoleRuntime("test", "test:scope", "test-model", "high", "default")
            for role in Role
        }
        for wid, cid, timestamp in (
            ("w_main", "last", 10),
            ("w_b", "shared", 20),
            ("w_a", "shared", 20),
            ("w_a", "first", 20),
            ("w_archived", "hidden", 100),
        ):
            store = self.application.state.turns.storage(wid)
            store.conversations.create(cid, runtimes=runtimes, title=f"Title {cid}")
            with store.turns.database.connect(write=True) as connection:
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (timestamp, cid),
                )
        service.registry.update("w_archived", state="archived")
        with patch.object(
            service,
            "prepare",
            side_effect=AssertionError("index must not prepare runtime"),
        ):
            response = await self.client.get("/api/agent/v1/conversations")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["sandbox_modes"], [sandbox.value for sandbox in Sandbox])
        records = body["conversations"]
        self.assertEqual(
            [(item["workspace_id"], item["id"]) for item in records],
            [
                ("w_a", "first"),
                ("w_a", "shared"),
                ("w_b", "shared"),
                ("w_main", "last"),
            ],
        )
        self.assertEqual([item["updated_at"] for item in records], [20, 20, 20, 10])
        for item in records:
            self.assertEqual(
                set(item), {"id", "title", "naming_state", "updated_at", "workspace_id"}
            )
            scoped = (
                await self.client.get(
                    f"{self.base}/{item['workspace_id']}/conversations"
                )
            ).json()["conversations"]
            self.assertIn(
                {key: value for key, value in item.items() if key != "workspace_id"},
                scoped,
            )
        archived = (
            await self.client.get(f"{self.base}/w_archived/conversations")
        ).json()
        self.assertEqual(archived["conversations"][0]["id"], "hidden")
        self.assertEqual(self.docker.calls, [])
