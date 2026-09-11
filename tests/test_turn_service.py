import asyncio
import json
import logging
import sqlite3
import unittest
from contextlib import aclosing
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import AgentMode, Role, Sandbox
from vibesim_agent.domain.turns import Outcome, TurnResult
from vibesim_agent.services.turn import TurnService, TurnStorage
from vibesim_agent.services.turn import _TURN_FAILURES
from vibesim_agent.storage.conversations import Conversations
from vibesim_agent.storage.database import Database
from vibesim_agent.storage.sessions import Sessions
from vibesim_agent.storage.turns import Turns


class Driver:
    def __init__(self):
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.requests = []
        self.closed = 0

    def validate(self, request):
        pass

    async def run(self, request):
        self.requests.append(request)
        try:
            yield {"kind": "role_start", "role": "assistant"}
            yield {"kind": "session", "role": "assistant", "session_id": "saved"}
            yield {"kind": "role_ready", "role": "assistant"}
            self.entered.set()
            await self.release.wait()
            yield TurnResult(Outcome.INPUT, "Which configuration?")
        finally:
            self.closed += 1


class TurnServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = Path(self.enterContext(TemporaryDirectory()))
        self.database = Database.create(directory / "workspace.sqlite")
        self.store = TurnStorage(Conversations(self.database), Sessions(self.database), Turns(self.database))
        runtime = RoleRuntime("test", "scope", "model", "high", "default")
        for conversation in ("c", "c2"):
            self.store.conversations.create(conversation, agent_mode=AgentMode.SINGLE,
                                             runtimes={Role.ASSISTANT: runtime})
        self.driver = Driver()
        self.service = TurnService(lambda _: self.store, self.driver,
                                   safe_interrupt_timeout=.1, logger=logging.getLogger("turn-test"))
        self.addAsyncCleanup(self.service.close)

    async def test_disconnect_resume_and_durable_terminal(self):
        handle = self.service.start("w", "c", "question")
        async with aclosing(self.service.stream("w", "c", handle.request.turn_id)) as stream:
            first = await anext(stream)
        await asyncio.wait_for(self.driver.entered.wait(), 1)
        self.assertFalse(handle.task.done())
        with self.assertRaisesRegex(ValueError, "active"):
            self.service.start("w", "c", "duplicate")
        self.driver.release.set()
        self.assertEqual((await self.service.wait(handle)).outcome, Outcome.INPUT)
        replay = [e async for e in self.service.stream("w", "c", handle.request.turn_id)]
        self.assertEqual(first, replay[0])
        self.assertEqual(replay[-1]["payload"]["outcome"], "request_user_input")
        self.assertEqual(sum(e["kind"] == "done" for e in replay), 1)
        self.assertEqual([m.content for m in self.store.conversations.messages("c")],
                         ["question", "Which configuration?"])
        next_handle = self.service.start("w", "c", "follow up")
        self.assertFalse(self.service.cancel("w", "c", handle.request.turn_id))
        await self.service.wait(next_handle)
        self.assertEqual(self.driver.requests[-1].sessions, {Role.ASSISTANT: "saved"})

    async def test_cancel_queued_turn_does_not_start_driver(self):
        first = self.service.start("w", "c", "first")
        await asyncio.wait_for(self.driver.entered.wait(), 1)
        second = self.service.start("w", "c2", "queued")
        self.assertTrue(self.service.cancel("w", "c2", second.request.turn_id))
        self.assertTrue(self.service.cancel("w", "c2", second.request.turn_id))
        result = await asyncio.wait_for(self.service.wait(second), 1)
        self.assertEqual(result.outcome, Outcome.CANCELLED)
        self.assertEqual(len(self.driver.requests), 1)
        self.assertEqual(len(self.store.conversations.messages("c2")), 2)
        self.service.cancel("w", "c", first.request.turn_id)
        await self.service.wait(first)
        self.assertEqual(self.store.conversations.get("c")["interrupted_role"], "assistant")

    async def test_cancel_waiter_does_not_cancel_background_execution(self):
        handle = self.service.start("w", "c", "question")
        waiter = asyncio.create_task(self.service.wait(handle))
        await asyncio.wait_for(self.driver.entered.wait(), 1)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertFalse(handle.task.done())
        self.driver.release.set()
        await self.service.wait(handle)

    async def test_failed_finalize_releases_memory_but_requires_durable_recovery(self):
        with self.database.connect(write=True) as connection:
            connection.execute("CREATE TRIGGER broken BEFORE INSERT ON turn_events "
                               "WHEN NEW.kind='done' BEGIN SELECT RAISE(ABORT, 'disk error'); END")
        handle = self.service.start("w", "c", "question")
        self.driver.release.set()
        with self.assertRaises(sqlite3.IntegrityError):
            await self.service.wait(handle)
        self.assertTrue(handle.finished)
        self.assertEqual(len(self.store.conversations.messages("c")), 1)
        with self.assertRaisesRegex(ValueError, "recovery"):
            self.service.start("w", "c", "next")
        with self.assertRaisesRegex(RuntimeError, "recovery"):
            _ = [event async for event in self.service.stream("w", "c", handle.request.turn_id)]

    async def test_restarted_service_stream_does_not_treat_running_turn_as_complete(self):
        self.store.turns.start("c", "interrupted", "question")
        self.store.turns.append_event("interrupted", "role_start", {"role": "assistant"})
        observed = []
        with self.assertRaisesRegex(RuntimeError, "recovery"):
            async for event in self.service.stream("w", "c", "interrupted"):
                observed.append(event)
        self.assertEqual([event["kind"] for event in observed], ["role_start"])

    async def test_slow_subscriber_drains_done_committed_during_older_batch(self):
        handle = self.service.start("w", "c", "question")
        await asyncio.wait_for(self.driver.entered.wait(), 1)
        async with aclosing(self.service.stream("w", "c", handle.request.turn_id)) as stream:
            first = await anext(stream)
            self.driver.release.set()
            await self.service.wait(handle)
            remaining = [event async for event in stream]
        replay = self.store.turns.events("c", handle.request.turn_id)
        self.assertEqual([first, *remaining], replay)
        self.assertEqual(remaining[-1]["kind"], "done")
        self.assertEqual(sum(event["kind"] == "done" for event in remaining), 1)

    async def test_queued_cancel_clears_previous_resume_role(self):
        first = self.service.start("w", "c", "occupy workspace")
        await asyncio.wait_for(self.driver.entered.wait(), 1)
        with self.database.connect(write=True) as connection:
            connection.execute("UPDATE conversations SET interrupted_role = 'assistant' WHERE id = 'c2'")
        second = self.service.start("w", "c2", "queued")
        self.service.cancel("w", "c2", second.request.turn_id)
        await self.service.wait(second)
        self.assertEqual(self.store.conversations.get("c2")["interrupted_role"], "")
        done = self.store.turns.events("c2", second.request.turn_id)[-1]["payload"]
        self.assertEqual(done["interrupted_role"], "")
        self.service.cancel("w", "c", first.request.turn_id)
        await self.service.wait(first)

    async def test_forced_unready_cancel_freezes_empty_landing_before_cleanup(self):
        started = asyncio.Event()
        holder = {}

        class UnreadyDriver(Driver):
            async def run(self, request):
                try:
                    yield {"kind": "role_start", "role": "assistant"}
                    started.set()
                    await asyncio.Event().wait()
                finally:
                    # Cleanup can observe a later role state; cancellation must
                    # retain the landing decision made before cleanup started.
                    holder["handle"].ready = True

        self.service.driver = UnreadyDriver()
        with self.database.connect(write=True) as connection:
            connection.execute("UPDATE conversations SET interrupted_role = 'assistant' WHERE id = 'c'")
        handle = self.service.start("w", "c", "question")
        holder["handle"] = handle
        await asyncio.wait_for(started.wait(), 1)
        self.service.cancel("w", "c", handle.request.turn_id)
        result = await asyncio.wait_for(self.service.wait(handle), 1)
        self.assertEqual(result.outcome, Outcome.CANCELLED)
        self.assertEqual(handle.cancel_resume_role, "")
        self.assertEqual(self.store.conversations.get("c")["interrupted_role"], "")

    async def test_driver_resume_metadata_and_configuration_survive_terminal_storage(self):
        class ReplyDriver(Driver):
            async def run(self, request):
                self.requests.append(request)
                yield TurnResult(Outcome.INPUT, "Need details.",
                                 metadata={"citations": [{"id": "citation"}]},
                                 resume_role=Role.IMPLEMENTER)

        driver = ReplyDriver()
        self.service.driver = driver
        runtime = self.store.conversations.runtimes("c")[Role.ASSISTANT]
        self.store.conversations.update_runtimes("c", {
            Role.ORCHESTRATOR: runtime, Role.IMPLEMENTER: runtime,
        })
        with self.database.connect(write=True) as connection:
            connection.execute("UPDATE conversations SET agent_mode = 'orchestrated', "
                               "sandbox = 'read-only', autonomous = 1, "
                               "peer_workspace = 'peer', prompt_fingerprint = 'fingerprint' WHERE id = 'c'")
        handle = self.service.start("w", "c", "question")
        result = await self.service.wait(handle)
        request = driver.requests[0]
        self.assertEqual(request.sandbox, Sandbox.READ_ONLY)
        self.assertTrue(request.autonomous)
        self.assertEqual(request.peer_workspace, "peer")
        self.assertEqual(request.prompt_fingerprint, "fingerprint")
        message = self.store.conversations.messages("c")[-1]
        done = self.store.turns.events("c", handle.request.turn_id)[-1]["payload"]
        self.assertEqual(message.metadata["citations"], done["citations"])
        self.assertEqual(message.metadata["interrupted_role"], "implementer")
        self.assertEqual(done["interrupted_role"], "implementer")
        self.assertEqual(self.store.conversations.get("c")["interrupted_role"], "implementer")
        self.assertEqual(result.resume_role, Role.IMPLEMENTER)
        self.assertEqual(result.metadata, message.metadata)
        self.assertEqual(result.text, done["text"])

    async def test_failure_keeps_resume_role_and_metadata_in_message_and_done(self):
        class FailingDriver(Driver):
            async def run(self, request):
                yield TurnResult(Outcome.FAILED, "Provider unavailable.",
                                 metadata={"failure": {"code": "upstream_unavailable"}})

        self.service.driver = FailingDriver()
        with self.database.connect(write=True) as connection:
            connection.execute("UPDATE conversations SET interrupted_role = 'assistant' WHERE id = 'c'")
        handle = self.service.start("w", "c", "question")
        result = await self.service.wait(handle)
        message = self.store.conversations.messages("c")[-1]
        done = self.store.turns.events("c", handle.request.turn_id)[-1]["payload"]
        self.assertEqual(done["failure"], {"code": "upstream_unavailable", "message": _TURN_FAILURES["upstream_unavailable"]})
        self.assertEqual(message.metadata["failure"], done["failure"])
        self.assertEqual(done["interrupted_role"], "assistant")
        self.assertEqual(self.store.conversations.get("c")["interrupted_role"], "assistant")
        self.assertEqual(result.resume_role, Role.ASSISTANT)
        self.assertEqual(result.metadata, message.metadata)

    async def test_repeated_stop_does_not_interrupt_driver_cleanup(self):
        entered = asyncio.Event()
        cleanup_started = asyncio.Event()
        finish_cleanup = asyncio.Event()
        cleaned = []

        class CleanupDriver(Driver):
            async def run(self, request):
                try:
                    yield {"kind": "role_start", "role": "assistant"}
                    yield {"kind": "role_ready", "role": "assistant"}
                    entered.set()
                    await asyncio.Event().wait()
                finally:
                    cleanup_started.set()
                    await finish_cleanup.wait()
                    cleaned.append(True)

        self.service.driver = CleanupDriver()
        handle = self.service.start("w", "c", "question")
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.service.cancel("w", "c", handle.request.turn_id)
            await asyncio.wait_for(cleanup_started.wait(), 1)
            self.service.cancel("w", "c", handle.request.turn_id)
            self.assertEqual(handle.task.cancelling(), 1)
        finally:
            finish_cleanup.set()
        result = await self.service.wait(handle)
        self.assertEqual(cleaned, [True])
        self.assertEqual(self.store.conversations.get("c")["interrupted_role"], "assistant")
        self.assertEqual(result.resume_role, Role.ASSISTANT)
        self.assertEqual(result.metadata, self.store.conversations.messages("c")[-1].metadata)
        self.assertEqual(result.metadata["interrupted_role"],
                         self.store.turns.events("c", handle.request.turn_id)[-1]["payload"]["interrupted_role"])

    async def test_driver_cleanup_failure_is_not_reported_as_successful_stop(self):
        entered = asyncio.Event()

        class FailedCleanupDriver(Driver):
            async def run(self, request):
                try:
                    yield {"kind": "role_start", "role": "assistant"}
                    yield {"kind": "role_ready", "role": "assistant"}
                    entered.set()
                    await asyncio.Event().wait()
                finally:
                    raise RuntimeError("remote process could not be stopped")

        self.service.driver = FailedCleanupDriver()
        handle = self.service.start("w", "c", "question")
        await asyncio.wait_for(entered.wait(), 1)
        self.service.cancel("w", "c", handle.request.turn_id)
        result = await self.service.wait(handle)
        self.assertEqual(result.outcome, Outcome.FAILED)
        self.assertNotEqual(result.text, "Stopped.")
        self.assertEqual(self.store.turns.get("c", handle.request.turn_id)["status"], "failed")
        self.assertEqual(self.store.turns.events("c", handle.request.turn_id)[-1]["payload"]["outcome"], "failed")
        failure = self.store.conversations.messages("c")[-1].metadata["failure"]
        self.assertEqual(failure, {"code": "agent_runtime_failure", "message": _TURN_FAILURES["agent_runtime_failure"]})

    async def test_history_projects_persisted_checkpoints_usage_and_job_interleaving(self):
        service_store = self.store
        emitted = [
            {"kind": "intermediate_output", "role": "assistant", "model": "model",
             "effort": "high", "level": "milestone", "text": "Checkpoint ready."},
            {"kind": "usage", "role": "assistant", "model": "model", "effort": "high",
             "duration_ms": 42, "tokens": {"read": 2, "prefill": 3, "output": 5}},
            {"kind": "decision", "action": "delegate", "task": "Inspect output"},
            {"kind": "implementer", "text": "Inspected."},
        ]

        class HistoryDriver(Driver):
            async def run(self, request):
                for index, event in enumerate(emitted):
                    yield event
                    if index == 0:
                        # Simulate a managed callback which persists independently
                        # of driver output, preserving its true position.
                        service_store.turns.append_event(request.turn_id, "job.ready", {
                            "jobId": "job", "jobKind": "simulation", "resourceId": "resource",
                        })
                yield TurnResult(Outcome.INPUT, "Which next step?")

        self.service.driver = HistoryDriver()
        handle = self.service.start("w", "c", "question")
        result = await self.service.wait(handle)
        message = self.store.conversations.messages("c")[-1]
        done = self.store.turns.events("c", handle.request.turn_id)[-1]["payload"]
        activity = message.metadata["activity"]
        self.assertEqual([entry["kind"] for entry in activity],
                         ["intermediate_output", "job", "usage", "decision", "implementer", "final"])
        self.assertEqual(activity[1]["workspaceId"], "w")
        self.assertEqual(activity[1]["jobId"], "job")
        self.assertEqual(activity[2], emitted[1])
        self.assertEqual(activity[-1], {"kind": "final", "text": result.text,
                                       "outcome": "request_user_input"})
        self.assertEqual(message.metadata["intermediate_outputs"],
                         [{key: value for key, value in emitted[0].items() if key != "kind"}])
        self.assertEqual(done["activity"], activity)
        self.assertEqual(result.metadata, message.metadata)
        self.assertIsNone(done["failure"])

    async def test_cancelled_history_keeps_cancelled_terminal_classification(self):
        handle = self.service.start("w", "c", "question")
        await asyncio.wait_for(self.driver.entered.wait(), 1)
        self.service.cancel("w", "c", handle.request.turn_id)
        result = await self.service.wait(handle)
        terminal = self.store.conversations.messages("c")[-1].metadata["activity"][-1]
        self.assertEqual(terminal, {"kind": "final", "text": "Stopped.", "outcome": "cancelled"})
        self.assertEqual(result.metadata["activity"][-1], terminal)

    async def test_transport_failure_normalizes_browser_failure_in_done_and_history(self):
        class ProviderFailureDriver(Driver):
            async def run(self, request):
                yield TurnResult(Outcome.FAILED, "Provider unavailable.",
                                 metadata={"failure": {"code": "upstream_unavailable", "status": 503}})

        self.service.driver = ProviderFailureDriver()
        handle = self.service.start("w", "c", "question")
        result = await self.service.wait(handle)
        message = self.store.conversations.messages("c")[-1]
        done = self.store.turns.events("c", handle.request.turn_id)[-1]["payload"]
        expected = {"code": "upstream_unavailable", "message": _TURN_FAILURES["upstream_unavailable"]}
        self.assertEqual(result.metadata["failure"], expected)
        self.assertEqual(message.metadata["failure"], expected)
        self.assertEqual(done["failure"], expected)
        self.assertEqual(message.metadata["activity"][-1], {"kind": "error", "text": result.text})

    def test_failure_messages_preserve_legacy_static_contract_without_importing_app(self):
        source = Path(__file__).parent / "fixtures/legacy_turn_failures.json"
        self.assertEqual(_TURN_FAILURES, json.loads(source.read_text())["failures"])

    async def test_unknown_failure_code_does_not_leak_into_structured_failure(self):
        class UnknownFailureDriver(Driver):
            async def run(self, request):
                yield TurnResult(Outcome.FAILED, "No result.", metadata={
                    "failure": {"code": "/private/credential", "message": "private detail"}
                })

        self.service.driver = UnknownFailureDriver()
        handle = self.service.start("w", "c", "question")
        result = await self.service.wait(handle)
        self.assertEqual(result.metadata["failure"], {
            "code": "agent_runtime_failure", "message": _TURN_FAILURES["agent_runtime_failure"]
        })
        self.assertNotIn("private", str(result.metadata["failure"]))
