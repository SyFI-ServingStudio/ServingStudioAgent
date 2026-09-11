import asyncio
import json
import sqlite3
import unittest
from unittest.mock import AsyncMock, patch

from tests import test_tools as fixtures
from vibesim_agent.domain.turns import Outcome
from vibesim_agent.runtime.container import OWNER_LABEL
from vibesim_agent.storage.jobs import Jobs


class ConversationDeletionTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ToolsTests.setUp
    asyncSetUp = fixtures.ToolsTests.asyncSetUp
    app = fixtures.ToolsTests.app
    providers = fixtures.ToolsTests.providers
    create = fixtures.ToolsTests.create

    def browser_path(self, cid):
        return f"/api/agent/v1/workspaces/w_main/conversations/{cid}"

    def runtime_path(self, cid):
        return self.state / "runtime" / cid

    def store(self):
        return self.application.state.turns.storage("w_main")

    async def test_browser_delete_active_turn_cascades_but_keeps_experiment_and_logs(
        self,
    ):
        await self.create()
        self.release.clear()
        turns = self.application.state.turns
        handle = turns.start("w_main", self.cid, "work")
        await asyncio.wait_for(self.entered.wait(), 3)
        jobs = Jobs(self.store().turns.database)
        jobs.create_simulation(
            conversation_id=self.cid,
            turn_id=handle.request.turn_id,
            role="assistant",
            job_id="j",
            experiment_id="e",
            experiment_path="experiment",
            event={"kind": "simulation.requested"},
        )
        logs = self.repo / "logs/experiment"
        logs.mkdir(parents=True)
        (logs / "result.txt").write_text("preserved")
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep").write_text("preserved")
        (self.runtime_path(self.cid) / "nested-link").symlink_to(
            outside, target_is_directory=True
        )
        owner = json.dumps(["test", "w_main", self.cid])
        self.docker.current = {
            "Id": "owned",
            "Config": {"Labels": {OWNER_LABEL: owner}},
        }
        response = await asyncio.wait_for(
            self.client.delete(self.browser_path(self.cid)), 3
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((await turns.wait(handle)).outcome, Outcome.CANCELLED)
        self.assertFalse(self.runtime_path(self.cid).exists())
        self.assertIsNone(self.store().conversations.get(self.cid))
        with self.store().turns.database.connect() as connection:
            for table in (
                "role_settings",
                "agent_sessions",
                "messages",
                "turns",
                "turn_events",
                "execution_jobs",
                "conversation_experiments",
            ):
                self.assertEqual(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0],
                    0,
                    table,
                )
        self.assertIsNotNone(jobs.get_experiment("e"))
        self.assertEqual((logs / "result.txt").read_text(), "preserved")
        self.assertEqual((outside / "keep").read_text(), "preserved")
        self.assertIn(["docker", "rm", "-f", "owned"], self.docker.calls)

    async def test_tools_delete_requires_token_and_removes_idle_conversation(self):
        path = await self.create()
        denied = await self.client.delete(path)
        self.assertEqual(denied.status_code, 401)
        self.assertIsNotNone(self.store().conversations.get(self.cid))
        removed = await self.client.delete(path, headers=self.headers)
        self.assertEqual(removed.status_code, 200, removed.text)
        self.assertIsNone(self.store().conversations.get(self.cid))
        repeated = await self.client.delete(path, headers=self.headers)
        self.assertEqual(repeated.status_code, 200)

    async def test_duplicate_delete_and_cancelled_waiter_keep_one_cleanup_and_block_send(
        self,
    ):
        await self.create()
        turns = self.application.state.turns
        entered, release = asyncio.Event(), asyncio.Event()
        cleanup_calls = []

        async def cleanup(wid, cid):
            cleanup_calls.append((wid, cid))
            entered.set()
            await release.wait()
            await self.application.state.runtime.cleanup(wid, cid)

        first = asyncio.create_task(turns.delete("w_main", self.cid, cleanup))
        self.addCleanup(release.set)
        await asyncio.wait_for(entered.wait(), 3)
        second = asyncio.create_task(turns.delete("w_main", self.cid, cleanup))
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        with self.assertRaisesRegex(ValueError, "delet"):
            turns.start("w_main", self.cid, "not admitted")
        self.assertFalse(second.done())
        release.set()
        await asyncio.wait_for(second, 3)
        self.assertEqual(cleanup_calls, [("w_main", self.cid)])
        self.assertIsNone(self.store().conversations.get(self.cid))

    async def test_cancelled_http_delete_waiter_does_not_cancel_cleanup(self):
        await self.create()
        entered, release = asyncio.Event(), asyncio.Event()
        runtime = self.application.state.runtime
        original = runtime._thread

        async def blocked(operation, *args):
            entered.set()
            await release.wait()
            return await original(operation, *args)

        self.addCleanup(release.set)
        with patch.object(runtime, "_thread", side_effect=blocked):
            deleting = asyncio.create_task(
                self.client.delete(self.browser_path(self.cid))
            )
            await asyncio.wait_for(entered.wait(), 3)
            deleting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await deleting
            self.assertIsNotNone(self.store().conversations.get(self.cid))
            release.set()
            await asyncio.wait_for(self.application.state.turns.close(), 3)
        self.assertIsNone(self.store().conversations.get(self.cid))

    async def test_queued_delete_waits_workspace_lock_without_deadlock(self):
        await self.create()
        cid = self.cid
        turns = self.application.state.turns
        turns.safe_interrupt_timeout = 0.01
        lock = turns._locks.setdefault("w_main", asyncio.Lock())
        await lock.acquire()
        self.addCleanup(lambda: lock.release() if lock.locked() else None)
        handle = turns.start("w_main", cid, "queued")
        await asyncio.sleep(0)
        deleting = asyncio.create_task(
            turns.delete("w_main", cid, self.application.state.runtime.cleanup)
        )
        result = await asyncio.wait_for(turns.wait(handle), 3)
        self.assertEqual(result.outcome, Outcome.CANCELLED)
        self.assertFalse(deleting.done())
        self.assertEqual(self.calls, [])
        lock.release()
        await asyncio.wait_for(deleting, 3)
        self.assertIsNone(self.store().conversations.get(cid))

    async def test_cleanup_failure_and_foreign_owner_preserve_home_and_database_then_retry(
        self,
    ):
        await self.create()
        root = self.runtime_path(self.cid)
        root.mkdir(parents=True)
        (root / "state").write_text("keep")
        runtime = self.application.state.runtime
        turns = self.application.state.turns
        self.docker.current = {
            "Id": "foreign",
            "Config": {"Labels": {OWNER_LABEL: "other"}},
        }
        with self.assertRaisesRegex(RuntimeError, "another owner"):
            await turns.delete("w_main", self.cid, runtime.cleanup)
        self.assertEqual((root / "state").read_text(), "keep")
        self.assertIsNotNone(self.store().conversations.get(self.cid))
        self.assertFalse(any(command[1] == "rm" for command in self.docker.calls))
        self.docker.current = {
            "Id": "owned",
            "Config": {
                "Labels": {OWNER_LABEL: json.dumps(["test", "w_main", self.cid])}
            },
        }
        self.docker.fail = "rm"
        with self.assertRaises(RuntimeError):
            await turns.delete("w_main", self.cid, runtime.cleanup)
        self.assertEqual((root / "state").read_text(), "keep")
        self.assertIsNotNone(self.store().conversations.get(self.cid))
        self.docker.fail = None
        await turns.delete("w_main", self.cid, runtime.cleanup)
        self.assertFalse(root.exists())
        self.assertIsNone(self.store().conversations.get(self.cid))

    async def test_shutdown_drains_pending_deletion(self):
        await self.create()
        entered, release = asyncio.Event(), asyncio.Event()
        turns = self.application.state.turns

        async def cleanup(wid, cid):
            entered.set()
            await release.wait()

        self.addCleanup(release.set)
        deleting = asyncio.create_task(turns.delete("w_main", self.cid, cleanup))
        await asyncio.wait_for(entered.wait(), 3)
        closing = asyncio.create_task(turns.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        release.set()
        await asyncio.wait_for(asyncio.gather(deleting, closing), 3)
        self.assertIsNone(self.store().conversations.get(self.cid))

    async def test_symlink_runtime_root_or_prefix_rejected_without_container_removal(
        self,
    ):
        await self.create()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep").write_text("preserved")
        prefix = self.state / "runtime"
        prefix.mkdir()
        root = prefix / self.cid
        root.symlink_to(outside, target_is_directory=True)
        for mode in ("root", "prefix"):
            with self.subTest(mode=mode):
                response = await self.client.delete(self.browser_path(self.cid))
                self.assertEqual(response.status_code, 409, response.text)
                self.assertIsNotNone(self.store().conversations.get(self.cid))
                self.assertEqual((outside / "keep").read_text(), "preserved")
                self.assertEqual(self.docker.calls, [])
            if mode == "root":
                root.unlink()
                prefix.rmdir()
                prefix.symlink_to(outside, target_is_directory=True)

    async def test_wrong_runtime_callback_cannot_remove_prefix_or_other_conversation(
        self,
    ):
        await self.create()
        runtime = self.application.state.runtime
        other = self.state / "runtime/other"
        other.mkdir(parents=True)
        (other / "keep").write_text("preserved")
        for destination in (self.state / "runtime", other):
            with (
                self.subTest(destination=destination),
                patch.object(runtime, "conversation_path", return_value=destination),
            ):
                with self.assertRaises(ValueError):
                    await runtime.cleanup("w_main", self.cid)
                self.assertEqual((other / "keep").read_text(), "preserved")
                self.assertEqual(self.docker.calls, [])

    async def test_database_delete_failure_after_cleanup_preserves_row_and_allows_retry(
        self,
    ):
        await self.create()
        root = self.runtime_path(self.cid)
        root.mkdir(parents=True)
        (root / "state").write_text("temporary")
        turns = self.application.state.turns
        with (
            patch.object(
                self.store().conversations,
                "delete",
                side_effect=sqlite3.OperationalError("database failure"),
            ),
            self.assertRaises(sqlite3.OperationalError),
        ):
            await turns.delete(
                "w_main", self.cid, self.application.state.runtime.cleanup
            )
        self.assertFalse(root.exists())
        self.assertIsNotNone(self.store().conversations.get(self.cid))
        await turns.delete("w_main", self.cid, self.application.state.runtime.cleanup)
        self.assertIsNone(self.store().conversations.get(self.cid))

    async def test_active_turn_finish_failure_blocks_cleanup_and_preserves_state(self):
        await self.create()
        self.release.clear()
        turns = self.application.state.turns
        handle = turns.start("w_main", self.cid, "work")
        await asyncio.wait_for(self.entered.wait(), 3)
        cleanup = AsyncMock(wraps=self.application.state.runtime.cleanup)
        with (
            patch.object(
                self.store().turns,
                "finish",
                side_effect=sqlite3.OperationalError("finish failed"),
            ),
            self.assertRaisesRegex(sqlite3.OperationalError, "finish failed"),
        ):
            await asyncio.wait_for(turns.delete("w_main", self.cid, cleanup), 3)
        cleanup.assert_not_awaited()
        self.assertTrue(self.runtime_path(self.cid).is_dir())
        self.assertIsNotNone(self.store().conversations.get(self.cid))
        self.assertEqual(
            self.store().turns.get(self.cid, handle.request.turn_id)["status"],
            "running",
        )
