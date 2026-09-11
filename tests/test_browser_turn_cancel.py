"""Cancelling a browser turn, including one that never got to start.

Turns are serialized per workspace, so a second conversation's turn can be
cancelled while it is still waiting for the first to finish. That cancellation
arrives before the turn has run a single line of its own body, which is exactly
the case where "the cleanup runs on the way out" stops being true.

The other half of the file is about *where* a running turn may be interrupted.
The server decides that, not the browser: a browser sees role events late and as
a replay, so it can be told a role is ready after the server has already left it.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend import app as app_module
from backend.store import Store, WorkspaceRegistry


def _store(temporary_directory: str) -> Store:
    main_dir = Path(temporary_directory) / "main"
    (main_dir / "logs").mkdir(parents=True)
    registry = WorkspaceRegistry(
        Path(temporary_directory) / "agent-workspaces",
        main_dir=main_dir,
    )
    return Store(registry)


async def _idle_task() -> None:
    """A stand-in for a turn that is running and will not end by itself."""
    await asyncio.sleep(3600)


def _frame(frame: str) -> dict | None:
    """The payload of a `done` frame, or None for anything else on the wire."""
    if not frame.startswith("event: done\n"):
        return None
    return json.loads(frame.split("data: ", 1)[1])


def _running(turn_id: str = "t") -> app_module.ActiveBrowserTurn:
    """A registered turn whose body is running, which is the normal case.

    `started` is set explicitly because it is what makes the turn cancellable at
    all: a task cancelled before its first step never enters its own body, so
    none of its cleanup runs.
    """
    active_turn = app_module.ActiveBrowserTurn(turn_id=turn_id)
    active_turn.started = True
    active_turn.task = asyncio.create_task(_idle_task())
    return active_turn


class QueuedTurnCancelTests(unittest.TestCase):
    def test_cancelling_a_queued_turn_releases_the_conversation(self) -> None:
        async def scenario() -> None:
            with TemporaryDirectory() as temporary_directory:
                store = _store(temporary_directory)
                store.create("w_main", "queued", "workspace-write")
                store.add_message("w_main", "queued", "user", "ask")
                store.start_turn("w_main", "queued", "turn_queued")

                active_turn = app_module.ActiveBrowserTurn(turn_id="turn_queued")
                app_module._active_browser_turns[("w_main", "queued")] = active_turn

                async def never_runs(*args: object, **kwargs: object):
                    raise AssertionError("the queued turn must not start")
                    yield  # pragma: no cover - generator marker

                # Another conversation in this workspace holds the turn lease.
                held = app_module._lock_for("w_main")
                await held.acquire()
                try:
                    with (
                        patch.object(app_module, "store", store),
                        patch.object(app_module, "run_turn", never_runs),
                        patch.object(app_module, "remove_managed_context"),
                        patch.object(app_module.capabilities, "revoke_turn"),
                    ):
                        task = asyncio.create_task(
                            app_module._run_browser_turn(
                                workspace_id="w_main",
                                cid="queued",
                                text="ask",
                                sandbox="workspace-write",
                                sessions={},
                                turn_id="turn_queued",
                                prompt_fingerprint="f",
                                autonomous=False,
                                analyzer_context=None,
                                active_turn=active_turn,
                            )
                        )
                        active_turn.task = task
                        # Let it reach the lock it will never get.
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())

                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    # Asserted here rather than after the cleanup below, which
                    # removes the registration itself and would pass whether or
                    # not the turn ever did. `finally` still runs if these fail.
                    #
                    # The browser is waiting on this stream. If it never
                    # finishes, the conversation reads as permanently working.
                    self.assertTrue(active_turn.finished)
                    # And the next message must not be refused as a conflict.
                    self.assertNotIn(
                        ("w_main", "queued"), app_module._active_browser_turns
                    )
                    self.assertEqual(store.list_running_turns("w_main"), [])
                finally:
                    held.release()
                    app_module._workspace_locks.pop("w_main", None)
                    app_module._active_browser_turns.pop(("w_main", "queued"), None)

                conversation = store.get("w_main", "queued")
                assert conversation is not None
                last = conversation["messages"][-1]
                self.assertEqual(last["role"], "assistant")
                self.assertEqual(last["content"], "Stopped.")
                # Said in the record, not left to be inferred from its shape: a
                # `final` step with no outcome means "answered" to a reader of
                # the stored form, so a stopped turn read back as one that
                # answered the user.
                self.assertEqual(
                    last["activity"][-1], {"kind": "final", "text": "Stopped.", "outcome": "cancelled"}
                )

        asyncio.run(scenario())

    def test_a_failed_final_write_still_releases_the_conversation(self) -> None:
        """Losing the last message is bad; losing the conversation is worse."""

        async def scenario() -> None:
            with TemporaryDirectory() as temporary_directory:
                store = _store(temporary_directory)
                store.create("w_main", "wedged", "workspace-write")
                store.start_turn("w_main", "wedged", "turn_wedged")

                active_turn = app_module.ActiveBrowserTurn(turn_id="turn_wedged")
                app_module._active_browser_turns[("w_main", "wedged")] = active_turn

                async def one_answer(*args: object, **kwargs: object):
                    yield {"kind": "final", "text": "done", "outcome": "final_answer"}

                def refuse(*args: object, **kwargs: object) -> None:
                    raise RuntimeError("disk full")

                try:
                    with (
                        patch.object(app_module, "store", store),
                        patch.object(app_module, "run_turn", one_answer),
                        patch.object(app_module, "remove_managed_context"),
                        patch.object(app_module.capabilities, "revoke_turn"),
                        patch.object(store, "add_message", refuse),
                    ):
                        await app_module._run_browser_turn(
                            workspace_id="w_main",
                            cid="wedged",
                            text="ask",
                            sandbox="workspace-write",
                            sessions={},
                            turn_id="turn_wedged",
                            prompt_fingerprint="f",
                            autonomous=False,
                            analyzer_context=None,
                            active_turn=active_turn,
                        )

                    self.assertTrue(active_turn.finished)
                    self.assertNotIn(
                        ("w_main", "wedged"), app_module._active_browser_turns
                    )
                    self.assertEqual(store.list_running_turns("w_main"), [])
                finally:
                    app_module._workspace_locks.pop("w_main", None)
                    app_module._active_browser_turns.pop(("w_main", "wedged"), None)

        asyncio.run(scenario())


    def test_one_failed_teardown_step_does_not_take_the_others_with_it(self) -> None:
        """Each thing a turn has to let go of is let go of on its own.

        Sharing one `except` meant the first failure ended the block: a store
        write that raised left an issued capability alive for its full lifetime
        and a managed-context file on disk, neither of which had anything to do
        with the failure.
        """

        async def scenario() -> None:
            with TemporaryDirectory() as temporary_directory:
                store = _store(temporary_directory)
                store.create("w_main", "c1", "workspace-write")
                store.start_turn("w_main", "c1", "t")

                active_turn = app_module.ActiveBrowserTurn(turn_id="t")
                app_module._active_browser_turns[("w_main", "c1")] = active_turn
                revoked: list[str] = []
                contexts: list[str] = []

                async def one_answer(*args: object, **kwargs: object):
                    yield {"kind": "final", "text": "done", "outcome": "final_answer"}

                def refuse(*args: object, **kwargs: object) -> None:
                    raise RuntimeError("the turn record would not close")

                try:
                    with (
                        patch.object(app_module, "store", store),
                        patch.object(app_module, "run_turn", one_answer),
                        patch.object(store, "finish_turn", refuse),
                        patch.object(
                            app_module,
                            "remove_managed_context",
                            lambda *args: contexts.append("removed"),
                        ),
                        patch.object(
                            app_module.capabilities,
                            "revoke_turn",
                            lambda *args: revoked.append("revoked"),
                        ),
                    ):
                        await app_module._run_browser_turn(
                            workspace_id="w_main",
                            cid="c1",
                            text="ask",
                            sandbox="workspace-write",
                            sessions={},
                            turn_id="t",
                            prompt_fingerprint="f",
                            autonomous=False,
                            analyzer_context=None,
                            active_turn=active_turn,
                        )

                    self.assertEqual(revoked, ["revoked"])
                    self.assertEqual(contexts, ["removed"])
                    self.assertTrue(active_turn.finished)
                    self.assertNotIn(("w_main", "c1"), app_module._active_browser_turns)
                finally:
                    app_module._workspace_locks.pop("w_main", None)
                    app_module._active_browser_turns.pop(("w_main", "c1"), None)

        asyncio.run(scenario())


class SafeInterruptTests(unittest.TestCase):
    """Where it is safe to interrupt is decided here, not in the browser."""

    def test_waits_for_a_role_that_has_started_but_produced_nothing(self) -> None:
        async def scenario() -> None:
            active_turn = _running()
            active_turn.current_role = "implementer"
            active_turn.role_ready = False

            cancelling = asyncio.create_task(app_module._cancel_when_safe(active_turn))
            await asyncio.sleep(0)
            # Between `role_start` and `role_ready` the rollout does not yet
            # hold the prompt carrying the handoff.
            self.assertFalse(cancelling.done())
            self.assertFalse(active_turn.task.cancelled())

            await active_turn.enter_role("role_ready", "implementer", "e")
            self.assertEqual(
                await asyncio.wait_for(cancelling, timeout=1), (True, "implementer")
            )
            self.assertTrue(active_turn.task.cancelling())

        asyncio.run(scenario())

    def test_does_not_let_a_wake_up_authorize_the_next_handoff(self) -> None:
        """Readiness is re-checked at the cancel, not remembered from the wake.

        A turn can become ready and move straight into the next role's handoff
        before the waiter is scheduled again. Treating the wake-up as permission
        would land the cancel in that next role, whose rollout does not hold its
        prompt yet — so the handoff would be lost and the next message would
        resume a role that never received its instructions.
        """

        async def scenario() -> None:
            active_turn = app_module.ActiveBrowserTurn(turn_id="t")
            active_turn.started = True
            active_turn.current_role = "planner"
            handed_off = asyncio.Event()
            ready = asyncio.Event()

            async def turn() -> None:
                await handed_off.wait()
                # Ready and gone, without ever yielding in between.
                await active_turn.enter_role("role_ready", "planner", "e")
                await active_turn.enter_role("role_start", "implementer", "e")
                await ready.wait()
                await active_turn.enter_role("role_ready", "implementer", "e")
                await asyncio.sleep(3600)

            active_turn.task = asyncio.create_task(turn())
            cancelling = asyncio.create_task(app_module._cancel_when_safe(active_turn))
            await asyncio.sleep(0)
            handed_off.set()
            await asyncio.sleep(0.05)

            # Still waiting: the role it was woken for is not the role the turn
            # is in any more.
            self.assertFalse(cancelling.done())
            self.assertFalse(active_turn.task.cancelling())

            ready.set()
            self.assertEqual(
                await asyncio.wait_for(cancelling, timeout=1), (True, "implementer")
            )

        asyncio.run(scenario())

    def test_reports_nothing_cancelled_when_the_turn_ends_on_its_own(self) -> None:
        async def scenario() -> None:
            active_turn = app_module.ActiveBrowserTurn(turn_id="t")
            active_turn.started = True
            ending = asyncio.Event()

            async def turn() -> None:
                await ending.wait()

            active_turn.task = asyncio.create_task(turn())
            active_turn.current_role = "implementer"

            cancelling = asyncio.create_task(app_module._cancel_when_safe(active_turn))
            await asyncio.sleep(0)
            ending.set()
            await active_turn.finish()
            # Nothing was interrupted, so there is no landing role to record and
            # `/cancel` must not claim it stopped anything.
            self.assertEqual(await asyncio.wait_for(cancelling, timeout=1), (False, ""))

        asyncio.run(scenario())

    def test_does_not_wait_when_no_role_is_mid_handoff(self) -> None:
        async def scenario() -> None:
            # Nothing has started yet: there is no handoff to lose, and no role
            # for the next message to resume.
            idle = _running()
            self.assertEqual(
                await asyncio.wait_for(app_module._cancel_when_safe(idle), timeout=1),
                (True, ""),
            )

            # And a role that has already produced output is interruptible.
            ready = _running()
            ready.current_role = "planner"
            ready.role_ready = True
            self.assertEqual(
                await asyncio.wait_for(app_module._cancel_when_safe(ready), timeout=1),
                (True, "planner"),
            )

        asyncio.run(scenario())

    def test_forces_the_interrupt_rather_than_waiting_forever(self) -> None:
        async def scenario() -> None:
            active_turn = _running()
            active_turn.current_role = "implementer"
            with patch.object(app_module, "SAFE_INTERRUPT_TIMEOUT_S", 0.01):
                cancelled, role = await asyncio.wait_for(
                    app_module._cancel_when_safe(active_turn), timeout=1
                )
            # Stopping late is a worse answer than stopping unsafely — but the
            # handoff really is lost, so nothing is offered to resume.
            self.assertEqual((cancelled, role), (True, ""))
            self.assertFalse(active_turn.role_ready)

        asyncio.run(scenario())


class CancelEndpointTests(unittest.TestCase):
    """What a Stop stops, and what it writes down — through the real turn."""

    def _cancel_a_running_turn(
        self,
        ready: bool,
        stale_role: str = "implementer",
        timeout: float = 30.0,
    ) -> tuple[dict, str, str, dict]:
        """Run a turn that stops inside `planner`, and report what was recorded.

        Returns the `/cancel` response, the resume role stored at the moment the
        turn's stream ended, the resume role once everything has settled, and
        the turn's own last frame. The second of those is the point of the
        test: a conversation is free to take another message as soon as its
        stream ends and its registration is gone, so anything decided about the
        *next* message has to be stored by then.
        """
        result: dict = {}
        at_stream_end = ""
        settled = ""
        done_frame: dict = {}

        async def scenario() -> None:
            nonlocal result, at_stream_end, settled, done_frame
            with TemporaryDirectory() as temporary_directory:
                store = _store(temporary_directory)
                store.create("w_main", "c1", "workspace-write")
                store.start_turn("w_main", "c1", "t")
                # Left over from an earlier turn that ended inside a role.
                store.set_interrupted_role("w_main", "c1", stale_role)

                inside = asyncio.Event()

                async def one_role(*args: object, **kwargs: object):
                    yield {"kind": "role_start", "role": "planner"}
                    if ready:
                        yield {"kind": "role_ready", "role": "planner"}
                    inside.set()
                    await asyncio.sleep(3600)
                    yield {"kind": "final", "text": "unreachable"}

                active_turn = app_module.ActiveBrowserTurn(turn_id="t")
                app_module._active_browser_turns[("w_main", "c1")] = active_turn
                try:
                    with (
                        patch.object(app_module, "store", store),
                        patch.object(app_module, "run_turn", one_role),
                        patch.object(app_module, "remove_managed_context"),
                        patch.object(app_module.capabilities, "revoke_turn"),
                        patch.object(app_module, "SAFE_INTERRUPT_TIMEOUT_S", timeout),
                    ):

                        async def watch() -> None:
                            nonlocal at_stream_end, done_frame
                            async for frame in active_turn.stream():
                                parsed = _frame(frame)
                                if parsed is not None:
                                    done_frame = parsed
                            at_stream_end = store.read_interrupted_role("w_main", "c1")

                        active_turn.task = asyncio.create_task(
                            app_module._run_browser_turn(
                                workspace_id="w_main",
                                cid="c1",
                                text="ask",
                                sandbox="workspace-write",
                                sessions={},
                                turn_id="t",
                                prompt_fingerprint="f",
                                autonomous=False,
                                analyzer_context=None,
                                active_turn=active_turn,
                            )
                        )
                        watching = asyncio.create_task(watch())
                        await asyncio.wait_for(inside.wait(), timeout=1)
                        result = await asyncio.wait_for(
                            app_module.cancel_message("w_main", "c1"), timeout=5
                        )
                        await asyncio.wait_for(watching, timeout=1)
                        settled = store.read_interrupted_role("w_main", "c1")
                finally:
                    app_module._workspace_locks.pop("w_main", None)
                    app_module._active_browser_turns.pop(("w_main", "c1"), None)

        asyncio.run(scenario())
        return result, at_stream_end, settled, done_frame

    def test_records_the_role_a_stop_landed_in(self) -> None:
        result, at_stream_end, settled, _ = self._cancel_a_running_turn(ready=True)
        self.assertEqual(result, {"cancelled": True, "interrupted_role": "planner"})
        # Already stored when the conversation became available again, not
        # written by the `/cancel` handler afterwards: in between, another
        # message could have started a turn against the previous role.
        self.assertEqual(at_stream_end, "planner")
        self.assertEqual(settled, "planner")

    def test_a_forced_cancel_clears_a_stale_resume_role(self) -> None:
        # Mid-handoff and staying there, so the wait times out and the interrupt
        # is forced. The handoff is lost either way, so there is nothing to
        # resume — and the role left over from the previous turn is not it.
        result, at_stream_end, settled, _ = self._cancel_a_running_turn(
            ready=False, timeout=0.01
        )
        self.assertEqual(result, {"cancelled": True, "interrupted_role": ""})
        self.assertEqual(at_stream_end, "")
        self.assertEqual(settled, "")

    def test_the_last_frame_says_the_turn_was_stopped(self) -> None:
        """The browser watching live must not be told this turn answered.

        Its last frame used to carry a null outcome, which every reader took
        for an answer, and `replying_role` — the role the turn would have
        replied as, not the one it was stopped in. That frame arrives after the
        `/cancel` response, so it overwrote the correct role with a wrong one.
        """
        _, _, settled, done_frame = self._cancel_a_running_turn(ready=True)
        self.assertEqual(done_frame["outcome"], "cancelled")
        self.assertEqual(done_frame["interrupted_role"], settled)
        # And the sentence the reader is left with names that same role. It is
        # the durable half of the record — the frame is gone on reload, this
        # text is not — so the two disagreeing would mean the transcript says
        # one role was stopped and the next message continues with another.
        self.assertEqual(done_frame["text"], f"Stopped while the {settled} was working.")

    def test_a_forced_stop_does_not_name_a_role_it_did_not_keep(self) -> None:
        # Forced mid-handoff: the handoff is lost, so there is no role to
        # resume. Naming the role it happened to be in would promise the reader
        # a thread that no longer exists.
        _, _, settled, done_frame = self._cancel_a_running_turn(ready=False, timeout=0.01)
        self.assertEqual(settled, "")
        self.assertEqual(done_frame["text"], "Stopped.")

    def test_reports_nothing_to_stop_when_the_turn_is_already_over(self) -> None:
        async def scenario() -> None:
            with TemporaryDirectory() as temporary_directory:
                store = _store(temporary_directory)
                store.create("w_main", "c1", "workspace-write")
                store.set_interrupted_role("w_main", "c1", "implementer")

                with patch.object(app_module, "store", store):
                    self.assertEqual(
                        await app_module.cancel_message("w_main", "c1"),
                        {"cancelled": False},
                    )
                # A Stop that stopped nothing must not rewrite where the
                # conversation resumes.
                self.assertEqual(
                    store.read_interrupted_role("w_main", "c1"), "implementer"
                )

        asyncio.run(scenario())

    def test_leaves_a_different_turn_alone_when_the_stop_names_one(self) -> None:
        """A Stop that names a turn must not be applied to whatever replaced it.

        `/cancel` addresses the conversation, and a browser's Stop can be
        answered after its own turn has ended — the response to its message was
        slow, or lost, and the retry that covers that race goes out later. By
        then another tab may have started something. Without the name on the
        request, that other turn is what gets stopped, and the person who
        started it is given no reason for it at all.
        """

        async def scenario() -> None:
            with TemporaryDirectory() as temporary_directory:
                store = _store(temporary_directory)
                store.create("w_main", "c1", "workspace-write")
                running = _running("t_second")
                app_module._active_browser_turns[("w_main", "c1")] = running
                try:
                    with patch.object(app_module, "store", store):
                        self.assertEqual(
                            await app_module.cancel_message(
                                "w_main", "c1", turn_id="t_first"
                            ),
                            {"cancelled": False, "stale": True},
                        )
                        self.assertFalse(running.cancel_requested)
                        self.assertFalse(running.task.done())
                        # And the same request without a name still means
                        # "whatever is running", which is what a reader pressing
                        # Stop means.
                        self.assertEqual(
                            await app_module.cancel_message("w_main", "c1"),
                            {"cancelled": True, "interrupted_role": ""},
                        )
                finally:
                    app_module._active_browser_turns.pop(("w_main", "c1"), None)

        asyncio.run(scenario())

    def test_stops_the_turn_a_stop_names_when_it_is_still_the_one_running(self) -> None:
        async def scenario() -> None:
            with TemporaryDirectory() as temporary_directory:
                store = _store(temporary_directory)
                store.create("w_main", "c1", "workspace-write")
                running = _running("t_only")
                app_module._active_browser_turns[("w_main", "c1")] = running
                try:
                    with patch.object(app_module, "store", store):
                        self.assertEqual(
                            await app_module.cancel_message(
                                "w_main", "c1", turn_id="t_only"
                            ),
                            {"cancelled": True, "interrupted_role": ""},
                        )
                finally:
                    app_module._active_browser_turns.pop(("w_main", "c1"), None)

        asyncio.run(scenario())

    def test_a_stream_says_which_turn_it_is(self) -> None:
        """The name a Stop uses has to reach the browser before the first frame.

        It is on the response head for that reason: the retry that needs it runs
        as soon as the response arrives, which can be long before any event has
        been read — that delay is the race it exists for.
        """
        active_turn = app_module.ActiveBrowserTurn(turn_id="t_named")
        response = app_module._turn_stream_response(active_turn)
        self.assertEqual(response.headers["X-Turn-Id"], "t_named")


class ConcurrentStopTests(unittest.TestCase):
    """Two tabs, one turn, one Stop."""

    def test_a_second_stop_joins_the_first_instead_of_cancelling_again(self) -> None:
        """A second `task.cancel()` does not stop the turn twice.

        It lands inside the first cancellation's cleanup — which is waiting for
        the runtime process to exit — and abandons it half way, while both
        requests report success and the process survives.
        """

        async def scenario() -> None:
            active_turn = _running()
            active_turn.current_role = "planner"
            active_turn.role_ready = True
            cancels = 0
            real_cancel = active_turn.task.cancel

            def counting_cancel(*args: object) -> bool:
                nonlocal cancels
                cancels += 1
                return real_cancel(*args)

            active_turn.task.cancel = counting_cancel  # type: ignore[method-assign]

            first, second = await asyncio.gather(
                app_module._cancel_when_safe(active_turn),
                app_module._cancel_when_safe(active_turn),
            )
            # Both callers are told the same thing about the same cancellation.
            self.assertEqual(first, (True, "planner"))
            self.assertEqual(second, (True, "planner"))
            self.assertEqual(cancels, 1)

        asyncio.run(scenario())

    def test_does_not_cancel_a_turn_that_has_not_begun(self) -> None:
        """A task cancelled before its first step never runs its own cleanup.

        No `finally`, so the stream is never ended and the registration never
        removed: the browser waits forever, every later message is refused as a
        conflict, and Stop reports there is nothing to stop. The window is
        microseconds wide and closes as soon as the canceller yields, so the
        cancel waits for it rather than racing it.
        """

        async def scenario() -> None:
            active_turn = app_module.ActiveBrowserTurn(turn_id="t")
            entered = asyncio.Event()

            async def turn() -> None:
                try:
                    await active_turn.begin()
                    entered.set()
                    await asyncio.sleep(3600)
                finally:
                    await active_turn.finish()

            # Created and cancelled without ever yielding in between, which is
            # exactly what a `/cancel` already in the ready queue does.
            active_turn.task = asyncio.create_task(turn())
            cancelled, role = await asyncio.wait_for(
                app_module._cancel_when_safe(active_turn), timeout=1
            )
            self.assertEqual((cancelled, role), (True, ""))
            self.assertTrue(entered.is_set())
            try:
                await asyncio.wait_for(active_turn.task, timeout=1)
            except asyncio.CancelledError:
                pass
            # The turn's own cleanup ran, so the conversation is usable again.
            self.assertTrue(active_turn.finished)

        asyncio.run(scenario())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
