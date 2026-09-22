import asyncio
import json
import sqlite3
import threading
import unittest
from unittest.mock import patch

import httpx

from tests import test_application as fixtures
from vibesim_agent.domain.roles import Role
from vibesim_agent.runtime.container import OWNER_LABEL
from vibesim_agent.storage.database import Database, SchemaMismatch
from vibesim_agent.storage.sessions import Session

RESTART_TEXT = (
    "This turn was interrupted when the conversation backend restarted. "
    "Completed Agent activity was recovered above, but no final answer was produced. "
    "Send `continue` to resume from the existing workspace and role sessions."
)


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ApplicationTests.setUp
    app = fixtures.ApplicationTests.app
    providers = fixtures.ApplicationTests.providers

    managed_workspace = fixtures.ApplicationTests.managed_workspace

    async def asyncSetUp(self):
        Database.create(self.state / "workspace.sqlite")
        # Both modes have to recover: `w_main` is external and runs on the host,
        # `w_copy` is a managed copy and still owns a container.
        self.managed_workspace()
        self.application = self.app()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.application), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.addCleanup(self.application.state.recovery.close)

    async def seed(self, wid="w_main", *, events=True, tid="orphan"):
        base = f"/api/agent/v1/workspaces/{wid}/conversations"
        response = await self.client.post(base, json={"agentMode": "single"})
        self.assertEqual(response.status_code, 200, response.text)
        cid = response.json()["id"]
        store = self.application.state.turns.storage(wid)
        store.turns.start(cid, tid, "unfinished")
        store.sessions.save(
            cid, Session(Role.ASSISTANT, "test", "test:scope", "resume")
        )
        with store.turns.database.connect(write=True) as connection:
            connection.execute(
                "UPDATE conversations SET interrupted_role='assistant' WHERE id=?",
                (cid,),
            )
        if events:
            store.turns.append_event(
                tid,
                "intermediate_output",
                {"kind": "intermediate_output", "text": "checked"},
            )
            store.turns.append_event(
                tid, "usage", {"kind": "usage", "tokens": {"input_tokens": 17}}
            )
            with store.turns.database.connect(write=True) as connection:
                connection.execute(
                    "UPDATE turn_events SET sequence=sequence+8 WHERE turn_id=?", (tid,)
                )
        home = self.state.parent / wid / "runtime" / cid
        home.mkdir(parents=True)
        (home / "session-file").write_bytes(b"persistent session")
        context = self.application.state.managed_context.path(wid, cid)
        context.parent.mkdir(parents=True)
        context.write_text("stale context")
        capability = self.application.state.capabilities.issue(
            workspace_id=wid, conversation_id=cid, turn_id=tid, role="assistant"
        )
        return cid, store, home, context, capability

    def raw_events(self, store):
        with store.turns.database.connect() as connection:
            return [
                tuple(row)
                for row in connection.execute(
                    "SELECT id,turn_id,sequence,kind,payload_json FROM turn_events ORDER BY id"
                )
            ]

    def owned(self, cid, *, wid="w_main"):
        self.docker.current = {
            "Id": "owned",
            "Config": {"Labels": {OWNER_LABEL: json.dumps(["test", wid, cid])}},
        }

    async def test_lifespan_recovers_history_without_mutating_events_sessions_or_home(
        self,
    ):
        cid, store, home, context, capability = await self.seed()
        original_events = self.raw_events(store)
        original_sessions = store.sessions.list(cid)
        before = store.conversations.get(cid)
        self.owned(cid)
        async with self.application.router.lifespan_context(self.application):
            self.assertEqual(store.turns.get(cid, "orphan")["status"], "interrupted")
            self.assertEqual(self.raw_events(store), original_events)
            self.assertEqual(store.sessions.list(cid), original_sessions)
            self.assertEqual(
                store.conversations.get(cid)["interrupted_role"], "assistant"
            )
            self.assertGreater(
                store.conversations.get(cid)["updated_at"], before["updated_at"]
            )
            self.assertEqual(
                (home / "session-file").read_bytes(), b"persistent session"
            )
            self.assertFalse(context.exists())
            self.assertIsNone(
                self.application.state.capabilities.authorize(capability.token)
            )
            # `w_main` is external. Startup recovery must not reach Docker for
            # it at all -- a host-only deployment may not have Docker installed,
            # and the container client does not survive its absence.
            self.assertEqual(self.docker.calls, [])
            response = await self.client.get(
                f"/api/agent/v1/workspaces/w_main/conversations/{cid}"
            )
            answer = response.json()["messages"][-1]
            self.assertEqual(answer["content"], RESTART_TEXT)
            self.assertEqual(
                answer["activity"][-1], {"kind": "error", "text": RESTART_TEXT}
            )
            self.assertEqual(answer["activity"][0]["kind"], "intermediate_output")
            self.assertEqual(answer["activity"][0]["text"], "checked")
            self.assertEqual(answer["activity"][1]["tokens"], {"input_tokens": 17})
            self.assertEqual(await self.application.state.recovery.recover(), 0)
            self.assertEqual(len(store.conversations.messages(cid)), 2)
            self.docker.current = None
            resumed = await self.client.post(
                f"/api/agent/v1/workspaces/w_main/conversations/{cid}/messages",
                json={"text": "continue"},
            )
            self.assertEqual(resumed.status_code, 200, resumed.text)
            self.assertEqual(self.calls[-1][0].session_id, "resume")

    async def test_archived_workspace_and_empty_events_are_recovered_but_done_is_untouched(
        self,
    ):
        archived = self.state.parent / "w_archived"
        archived.mkdir()
        descriptor = json.loads((self.state / "workspace.json").read_text())
        descriptor.update(workspace_id="w_archived", display_name="Archived")
        (archived / "workspace.json").write_text(json.dumps(descriptor))
        Database.create(archived / "workspace.sqlite")
        cid, store, _, _, _ = await self.seed("w_archived", events=False)
        descriptor["state"] = "archived"
        (archived / "workspace.json").write_text(json.dumps(descriptor))
        main_cid, main, _, _, _ = await self.seed(tid="completed")
        main.turns.finish(main_cid, "completed", text="finished", status="complete")
        before = main.conversations.messages(main_cid)
        self.assertEqual(await self.application.state.recovery.recover(), 1)
        self.assertEqual(main.conversations.messages(main_cid), before)
        self.assertEqual(store.turns.get(cid, "orphan")["status"], "interrupted")
        self.assertEqual(store.turns.events(cid, "orphan"), [])
        self.assertEqual(
            store.conversations.messages(cid)[-1].metadata["activity"],
            [{"kind": "error", "text": RESTART_TEXT}],
        )

    async def test_terminal_insert_abort_rolls_back_and_can_retry(self):
        cid, store, home, _, _ = await self.seed()
        before = (
            store.conversations.get(cid),
            store.conversations.messages(cid),
            self.raw_events(store),
        )
        with store.turns.database.connect(write=True) as connection:
            connection.execute("""CREATE TRIGGER fail_recovery BEFORE INSERT ON messages
                WHEN NEW.role='assistant' BEGIN SELECT RAISE(ABORT, 'injected'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            await self.application.state.recovery.recover()
        self.assertEqual(store.turns.get(cid, "orphan")["status"], "running")
        self.assertEqual(
            (
                store.conversations.get(cid),
                store.conversations.messages(cid),
                self.raw_events(store),
            ),
            before,
        )
        self.assertTrue(home.exists())
        with store.turns.database.connect(write=True) as connection:
            connection.execute("DROP TRIGGER fail_recovery")
        self.assertEqual(await self.application.state.recovery.recover(), 1)

    async def test_a_managed_workspace_still_removes_its_container(self):
        cid, _, _, _, _ = await self.seed("w_copy")
        self.owned(cid, wid="w_copy")
        async with self.application.router.lifespan_context(self.application):
            self.assertIn(["docker", "rm", "-f", "owned"], self.docker.calls)

    async def test_foreign_owner_and_remove_failure_block_startup_until_retry(self):
        cid, store, home, _, _ = await self.seed("w_copy")
        for foreign in (True, False):
            self.owned("someone-else" if foreign else cid, wid="w_copy")
            self.docker.fail = None if foreign else "rm"
            yielded = False
            with self.assertRaises(RuntimeError):
                async with self.application.router.lifespan_context(self.application):
                    yielded = True
            self.assertFalse(yielded)
            self.assertEqual(store.turns.get(cid, "orphan")["status"], "running")
            self.assertEqual(len(store.conversations.messages(cid)), 1)
            self.assertTrue(home.exists())
        self.docker.fail = None
        self.assertEqual(await self.application.state.recovery.recover(), 1)

    async def test_exclusive_empty_startup_lock_blocks_second_backend_without_mutation(
        self,
    ):
        first = self.application.state.recovery
        self.assertEqual(await first.recover(), 0)
        cid, store, _, context, capability = await self.seed()
        second = self.app()
        self.addCleanup(second.state.recovery.close)
        calls = list(self.docker.calls)
        yielded = False
        with self.assertRaisesRegex(RuntimeError, "owned by another backend"):
            async with second.router.lifespan_context(second):
                yielded = True
        self.assertFalse(yielded)
        self.assertEqual(self.docker.calls, calls)
        self.assertTrue(context.exists())
        self.assertIsNotNone(
            self.application.state.capabilities.authorize(capability.token)
        )
        self.assertEqual(store.turns.get(cid, "orphan")["status"], "running")
        self.assertEqual(await first.recover(), 0)
        first.close()
        self.assertEqual(await second.state.recovery.recover(), 1)

    async def test_interrupt_owner_idempotency_and_running_order(self):
        cid, store, _, _, _ = await self.seed(tid="z")
        store.turns.start(cid, "a", "another unfinished turn")
        with store.turns.database.connect(write=True) as connection:
            connection.execute("UPDATE turns SET created_at=1")
        self.assertEqual([row["id"] for row in store.turns.running()], ["a", "z"])
        self.assertFalse(
            store.turns.interrupt("wrong-owner", "a", text="recovered", metadata={})
        )
        self.assertTrue(store.turns.interrupt(cid, "a", text="recovered", metadata={}))
        self.assertFalse(store.turns.interrupt(cid, "a", text="duplicate", metadata={}))
        self.assertEqual([row["id"] for row in store.turns.running()], ["z"])
        self.assertEqual(
            [m.content for m in store.conversations.messages(cid)].count("recovered"), 1
        )

    async def test_cancelled_recovery_holds_lock_until_cleanup_thread_finishes(self):
        cid, store, _, _, _ = await self.seed()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        second = self.app()
        self.addCleanup(second.state.recovery.close)

        def slow_remove(*args):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test cleanup thread was not released")

        with patch.object(
            self.application.state.runtime, "_release", side_effect=slow_remove
        ):
            recovering = asyncio.create_task(self.application.state.recovery.recover())
            try:
                async with asyncio.timeout(2):
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                recovering.cancel()
                await asyncio.sleep(0)
                self.assertFalse(recovering.done())
                with self.assertRaisesRegex(RuntimeError, "owned by another backend"):
                    await second.state.recovery.recover()
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(recovering, 3)
        self.assertEqual(store.turns.get(cid, "orphan")["status"], "running")
        self.assertEqual(await second.state.recovery.recover(), 1)

    async def test_symlink_runtime_rejection_preserves_foreign_context(self):
        cid, store, home, context, _ = await self.seed()
        saved = home.with_name("original-home")
        home.rename(saved)
        foreign = home.with_name("other-conversation")
        foreign_context = foreign / context.relative_to(home)
        foreign_context.parent.mkdir(parents=True)
        foreign_context.write_text("another conversation context")
        home.symlink_to(foreign, target_is_directory=True)
        with self.assertRaises(ValueError):
            await self.application.state.recovery.recover()
        self.assertEqual(foreign_context.read_text(), "another conversation context")
        self.assertEqual(store.turns.get(cid, "orphan")["status"], "running")
        self.assertFalse(any(call[1] == "rm" for call in self.docker.calls))
        home.unlink()
        saved.rename(home)
        self.assertEqual(await self.application.state.recovery.recover(), 1)

    async def test_bad_archived_database_preflight_prevents_all_cleanup(self):
        cid, store, _, context, capability = await self.seed()
        archived = self.state.parent / "w_broken"
        archived.mkdir()
        descriptor = json.loads((self.state / "workspace.json").read_text())
        descriptor.update(workspace_id="w_broken", state="archived")
        (archived / "workspace.json").write_text(json.dumps(descriptor))
        database_path = archived / "workspace.sqlite"
        with sqlite3.connect(database_path) as connection:
            connection.execute("CREATE TABLE legacy(value TEXT)")
        before = database_path.read_bytes()
        with self.assertRaises(SchemaMismatch):
            await self.application.state.recovery.recover()
        self.assertEqual(database_path.read_bytes(), before)
        self.assertEqual(self.docker.calls, [])
        self.assertEqual(context.read_text(), "stale context")
        self.assertIsNotNone(
            self.application.state.capabilities.authorize(capability.token)
        )
        self.assertEqual(store.turns.get(cid, "orphan")["status"], "running")
        self.assertEqual(len(store.conversations.messages(cid)), 1)
        database_path.unlink()
        Database.create(database_path)
        self.assertEqual(await self.application.state.recovery.recover(), 1)
