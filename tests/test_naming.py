import asyncio
import sqlite3
import unittest
from unittest.mock import patch

from pydantic import SecretStr
from tests import test_workspace_http as fixtures
from vibesim_agent.domain.roles import AgentMode
from vibesim_agent.domain.turns import Outcome, TurnInput, TurnResult
from vibesim_agent.services.name_generator import GeneratedNames


class NamingTests(unittest.IsolatedAsyncioTestCase):
    app = fixtures.WorkspaceHttpTests.app
    providers = fixtures.WorkspaceHttpTests.providers
    git = fixtures.WorkspaceHttpTests.git

    def setUp(self):
        fixtures.WorkspaceHttpTests.setUp(self)
        self.settings = self.settings.model_copy(
            update={"secrets": {"OPENROUTER_API_KEY": SecretStr("test-key")}}
        )
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.naming_calls = []
        self.fail_naming = False

    async def asyncSetUp(self):
        await fixtures.WorkspaceHttpTests.asyncSetUp(self)
        self.naming = self.application.state.naming
        self.addAsyncCleanup(self.naming.close)

        async def generate(user_message, final_answer):
            self.naming_calls.append((user_message, final_answer))
            self.entered.set()
            await self.release.wait()
            if self.fail_naming:
                raise RuntimeError("private naming detail")
            return GeneratedNames(
                workspace_name="Workload Study", conversation_title="Compare Batches"
            )

        self.enterContext(
            patch.object(self.naming.generator, "generate", side_effect=generate)
        )
        workspace = await self.client.post(
            self.base, json={"displayName": "New workspace", "autoName": True}
        )
        self.wid = workspace.json()["workspace_id"]
        created = await self.client.post(
            f"{self.base}/{self.wid}/conversations", json={"agentMode": "single"}
        )
        self.cid = created.json()["id"]
        self.path = f"{self.base}/{self.wid}/conversations/{self.cid}"
        self.store = self.application.state.turns.storage(self.wid)

    async def asyncTearDown(self):
        self.release.set()

    async def send(self, text="Compare batch sizes"):
        return await self.client.post(
            f"/api/agent/v1/tools/workspaces/{self.wid}/conversations/{self.cid}/messages",
            headers={"Authorization": "Bearer token"},
            json={"text": text},
        )

    async def drain_current(self):
        await asyncio.gather(
            *(ticket.task for ticket in list(self.naming._pending.values()))
        )
        await asyncio.sleep(0)

    def request(self):
        return TurnInput(
            self.wid, self.cid, "ticket", "question", AgentMode.SINGLE, {}, {}, ""
        )

    async def test_response_does_not_wait_and_durable_done_matches_tools_flag(self):
        response = await asyncio.wait_for(self.send(), 3)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["naming_scheduled"])
        await asyncio.wait_for(self.entered.wait(), 3)
        self.assertIsNone(self.store.turns.active(self.cid))
        events = self.store.turns.events(self.cid, result["turn_id"])
        self.assertTrue(events[-1]["payload"]["naming_scheduled"])
        self.assertEqual(
            self.application.state.workspaces.get(self.wid)["naming_state"], "pending"
        )
        changed = await self.client.patch(
            f"{self.base}/{self.wid}", json={"displayName": "Manual Name"}
        )
        self.assertEqual(changed.status_code, 200)
        self.release.set()
        await self.drain_current()
        self.assertEqual(
            self.application.state.workspaces.get(self.wid)["display_name"],
            "Manual Name",
        )
        self.assertEqual(
            self.store.conversations.get(self.cid)["title"], "Compare Batches"
        )
        self.assertEqual(
            self.store.conversations.get(self.cid)["naming_state"], "generated"
        )
        repeated = await self.send("another question")
        self.assertFalse(repeated.json()["naming_scheduled"])
        self.assertEqual(len(self.naming_calls), 1)

    async def test_manual_title_survives_naming_already_in_flight(self):
        response = await asyncio.wait_for(self.send(), 3)
        self.assertTrue(response.json()["naming_scheduled"])
        await asyncio.wait_for(self.entered.wait(), 3)
        before = self.store.conversations.get(self.cid)
        renamed = await self.client.patch(self.path, json={"title": "  My study  "})
        self.assertEqual(renamed.status_code, 200, renamed.text)
        self.assertEqual(renamed.json()["title"], "My study")
        self.release.set()
        await self.drain_current()
        current = self.store.conversations.get(self.cid)
        self.assertEqual(current["title"], "My study")
        self.assertEqual(current["naming_state"], "manual")
        self.assertEqual(current["updated_at"], before["updated_at"])
        # The workspace was still pending, so it takes its generated name.
        self.assertEqual(
            self.application.state.workspaces.get(self.wid)["display_name"],
            "Workload Study",
        )

    async def test_rename_rejects_blank_titles_and_unknown_conversations(self):
        for body in ({"title": ""}, {}):
            response = await self.client.patch(self.path, json=body)
            self.assertEqual(response.status_code, 422, response.text)
        blank = await self.client.patch(self.path, json={"title": "   "})
        self.assertEqual(blank.status_code, 400, blank.text)
        missing = await self.client.patch(
            f"{self.base}/{self.wid}/conversations/missing", json={"title": "Name"}
        )
        self.assertEqual(missing.status_code, 404, missing.text)
        self.assertEqual(self.store.conversations.get(self.cid)["title"], "New chat")

    async def test_terminal_database_failure_aborts_unstarted_naming(self):
        with (
            patch.object(
                self.store.turns,
                "finish",
                side_effect=sqlite3.OperationalError("finish failed"),
            ),
            self.assertRaisesRegex(sqlite3.OperationalError, "finish failed"),
        ):
            await self.send()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(self.naming_calls, [])
        self.assertEqual(self.naming._pending, {})
        self.assertEqual(
            self.store.conversations.get(self.cid)["naming_state"], "pending"
        )

    async def test_abort_before_first_task_step_releases_key_for_retry(self):
        request, result = self.request(), TurnResult(Outcome.ANSWER, "answer")
        ticket = self.naming.prepare(request, result)
        self.assertIsNotNone(ticket)
        ticket.abort()
        await asyncio.gather(ticket.task, return_exceptions=True)
        self.assertEqual(self.naming._pending, {})
        self.assertEqual(self.naming_calls, [])
        retry = self.naming.prepare(request, result)
        self.assertIsNotNone(retry)
        self.release.set()
        retry.commit()
        await retry.task
        self.assertEqual(len(self.naming_calls), 1)

    async def test_generation_failure_is_retryable_and_duplicate_pending_turn_is_not_scheduled(
        self,
    ):
        first = await self.send()
        self.assertTrue(first.json()["naming_scheduled"])
        await asyncio.wait_for(self.entered.wait(), 3)
        second = await self.send("while naming is pending")
        self.assertFalse(second.json()["naming_scheduled"])
        self.fail_naming = True
        self.release.set()
        await self.drain_current()
        self.assertEqual(
            self.store.conversations.get(self.cid)["naming_state"], "pending"
        )
        self.fail_naming = False
        retry = await self.send("retry naming")
        self.assertTrue(retry.json()["naming_scheduled"])
        await self.drain_current()
        self.assertEqual(
            self.store.conversations.get(self.cid)["title"], "Compare Batches"
        )

    async def test_deleted_conversation_does_not_rename_workspace(self):
        await self.send()
        await asyncio.wait_for(self.entered.wait(), 3)
        removed = await self.client.delete(self.path)
        self.assertEqual(removed.status_code, 200, removed.text)
        self.release.set()
        await self.drain_current()
        self.assertIsNone(self.store.conversations.get(self.cid))
        self.assertEqual(
            self.application.state.workspaces.get(self.wid)["naming_state"], "pending"
        )

    async def test_prepare_errors_and_non_successes_do_not_affect_turn_or_schedule(
        self,
    ):
        with patch.object(
            self.naming.registry, "get", side_effect=OSError("naming lookup failed")
        ):
            self.assertIsNone(
                self.naming.prepare(
                    self.request(), TurnResult(Outcome.ANSWER, "answer")
                )
            )
        for outcome in (Outcome.FAILED, Outcome.CANCELLED):
            self.assertIsNone(
                self.naming.prepare(
                    self.request(), TurnResult(outcome, "not a successful answer")
                )
            )
        with patch.object(
            self.naming, "conversations", side_effect=OSError("naming read failed")
        ):
            response = await self.send()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["ok"])
        self.assertFalse(response.json()["naming_scheduled"])

    async def test_close_cancels_uncommitted_gate(self):
        ticket = self.naming.prepare(
            self.request(), TurnResult(Outcome.ANSWER, "answer")
        )
        await asyncio.wait_for(self.naming.close(), 3)
        self.assertTrue(ticket.task.cancelled())
        self.assertEqual(self.naming_calls, [])
        self.assertIsNone(
            self.naming.prepare(self.request(), TurnResult(Outcome.ANSWER, "answer"))
        )

    async def test_cancelled_close_waiter_still_drains_committed_task(self):
        await self.send()
        await asyncio.wait_for(self.entered.wait(), 3)
        closing = asyncio.create_task(self.naming.close())
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        self.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(closing, 3)
        self.assertEqual(self.naming._pending, {})
        self.assertEqual(
            self.store.conversations.get(self.cid)["naming_state"], "generated"
        )

    async def test_eval_skips_naming_even_with_key_and_pending_conversation(self):
        response = await self.client.post(
            "/api/agent/v1/tools/eval",
            headers={"Authorization": "Bearer token"},
            json={"prompt": "evaluate", "agent_mode": "single", "keep_container": True},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["ok"])
        self.assertNotIn("naming_scheduled", response.json())
        self.assertEqual(self.naming_calls, [])
