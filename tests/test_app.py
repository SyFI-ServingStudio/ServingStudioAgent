from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, AsyncIterator
from unittest import mock

from fastapi import HTTPException

from backend import app as app_module
from backend.app import ActiveBrowserTurn, _run_browser_turn, list_codex_backends
from backend.codex_runtime.config import ALL_CODEX_ROLES
from backend.store import Store, WorkspaceRegistry


class CodexBackendCatalogTest(unittest.TestCase):
    def test_defaults_cover_every_role_not_just_the_default_mode(self) -> None:
        """The picker offers `agent_mode` before the first message, so it needs a
        default for `assistant` too — a role the default mode never runs."""
        defaults = list_codex_backends()["defaults"]

        self.assertEqual(sorted(defaults), sorted(ALL_CODEX_ROLES))
        for role, runtime in defaults.items():
            self.assertEqual(
                sorted(runtime), ["effort", "model", "serviceTier"], msg=role
            )


class SavedTurnActivityTest(unittest.IsolatedAsyncioTestCase):
    async def test_an_experiment_card_is_saved_where_it_happened(self) -> None:
        """The live stream interleaves experiments with the narration around
        them, on purpose: that narration is what gives an experiment its
        context. Saving them collected at the end instead made every card move
        on the turn's last frame."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            main_dir = root / "main"
            (main_dir / "logs").mkdir(parents=True)
            store = Store(
                WorkspaceRegistry(root / "agent-workspaces", main_dir=main_dir)
            )
            store.create("w_main", "conversation", "workspace-write")
            store.start_turn("w_main", "conversation", "turn")

            async def fake_run_turn(*_args: Any, **_kwargs: Any) -> AsyncIterator[dict]:
                yield {
                    "kind": "intermediate_output",
                    "role": "orchestrator",
                    "text": "launching the sweep",
                }
                # Written the way the managed-run callbacks write it: from
                # outside this stream, straight into the durable event log.
                store.append_turn_event(
                    "w_main",
                    "turn",
                    "simulation.requested",
                    {"experimentId": "exp1", "experimentPath": "logs/exp1"},
                )
                yield {
                    "kind": "intermediate_output",
                    "role": "orchestrator",
                    "text": "reading the results",
                }
                yield {"kind": "final", "text": "Done.", "outcome": "final_answer"}

            active_turn = ActiveBrowserTurn(turn_id="turn")
            with (
                mock.patch.object(app_module, "store", store),
                mock.patch.object(app_module, "run_turn", fake_run_turn),
                mock.patch.object(
                    app_module, "schedule_auto_naming", return_value=False
                ),
            ):
                await _run_browser_turn(
                    workspace_id="w_main",
                    cid="conversation",
                    text="run the sweep",
                    sandbox="workspace-write",
                    sessions={},
                    turn_id="turn",
                    prompt_fingerprint="",
                    autonomous=False,
                    analyzer_context=None,
                    active_turn=active_turn,
                )

            saved = store.get("w_main", "conversation")["messages"][-1]
            self.assertEqual(
                [entry["kind"] for entry in saved["activity"]],
                ["intermediate_output", "job", "intermediate_output", "final"],
            )


class NextMessageTargetTest(unittest.IsolatedAsyncioTestCase):
    """Who the user's next message reaches, decided by how the turn ended."""

    async def _end_turn_with(self, final_event: dict[str, Any]) -> str:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            main_dir = root / "main"
            (main_dir / "logs").mkdir(parents=True)
            store = Store(
                WorkspaceRegistry(root / "agent-workspaces", main_dir=main_dir)
            )
            store.create("w_main", "conversation", "workspace-write")
            store.start_turn("w_main", "conversation", "turn")
            store.set_interrupted_role("w_main", "conversation", "implementer")

            async def fake_run_turn(*_args: Any, **_kwargs: Any) -> AsyncIterator[dict]:
                yield final_event

            with (
                mock.patch.object(app_module, "store", store),
                mock.patch.object(app_module, "run_turn", fake_run_turn),
                mock.patch.object(
                    app_module, "schedule_auto_naming", return_value=False
                ),
            ):
                await _run_browser_turn(
                    workspace_id="w_main",
                    cid="conversation",
                    text="what is the task",
                    sandbox="workspace-write",
                    sessions={},
                    turn_id="turn",
                    prompt_fingerprint="",
                    autonomous=False,
                    analyzer_context=None,
                    active_turn=ActiveBrowserTurn(turn_id="turn"),
                )
            return str(store.get("w_main", "conversation")["interrupted_role"])

    async def test_a_direct_reply_keeps_the_conversation_with_its_author(
        self,
    ) -> None:
        """Asking the implementer a second question is as natural as the first,
        so the target survives the turn that answered the first."""
        self.assertEqual(
            await self._end_turn_with(
                {
                    "kind": "final",
                    "text": "Reprofiling the GEMM.",
                    "outcome": "final_answer",
                    "role": "implementer",
                }
            ),
            "implementer",
        )

    async def test_any_other_ending_returns_to_the_driving_role(self) -> None:
        """A turn that ends through the driver has handed the conversation back;
        leaving the old target set would reroute every message after it."""
        self.assertEqual(
            await self._end_turn_with(
                {"kind": "final", "text": "Done.", "outcome": "final_answer"}
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()


class TurnReplayTest(unittest.IsolatedAsyncioTestCase):
    """B4: drive the Agent UI from recorded history instead of a fake backend.

    A hand-written fake would drift from what the real backend emits, and the
    drift would be invisible — the fake would keep passing. Replaying stored
    `turn_events` cannot drift, because it is the same bytes the live turn
    published.
    """

    @staticmethod
    def _store(root: Path) -> Store:
        main_dir = root / "main"
        (main_dir / "logs").mkdir(parents=True)
        return Store(WorkspaceRegistry(root / "agent-workspaces", main_dir=main_dir))

    async def _collect(self, response: Any) -> list[str]:
        chunks: list[str] = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return chunks

    async def test_replays_recorded_events_in_order_without_changing_them(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            store = self._store(Path(temporary_directory))
            store.create("w_main", "conversation", "workspace-write")
            store.start_turn("w_main", "conversation", "turn-a")
            for index, kind in enumerate(["role_start", "delta", "final"]):
                store.append_turn_event(
                    "w_main", "turn-a", kind, {"text": f"event-{index}"}
                )
            store.finish_turn("w_main", "turn-a", "complete")

            with mock.patch.object(app_module, "store", store):
                index = app_module.list_conversation_turns("w_main", "conversation")
                response = await app_module.replay_turn(
                    "w_main", "conversation", "turn-a"
                )
                chunks = await self._collect(response)

            self.assertEqual(
                index["turns"],
                [
                    {
                        "turn_id": "turn-a",
                        "status": "complete",
                        "created_at": mock.ANY,
                        "updated_at": mock.ANY,
                        "event_count": 3,
                    }
                ],
            )
            self.assertEqual(response.media_type, "text/event-stream")
            self.assertEqual(
                chunks,
                [
                    'event: role_start\ndata: {"text": "event-0"}\n\n',
                    'event: delta\ndata: {"text": "event-1"}\n\n',
                    'event: final\ndata: {"text": "event-2"}\n\n',
                ],
            )

    async def test_replay_writes_nothing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            store = self._store(Path(temporary_directory))
            store.create("w_main", "conversation", "workspace-write")
            store.start_turn("w_main", "conversation", "turn-a")
            store.append_turn_event("w_main", "turn-a", "final", {"text": "done"})
            store.finish_turn("w_main", "turn-a", "complete")
            before = store.get("w_main", "conversation")

            with mock.patch.object(app_module, "store", store):
                response = await app_module.replay_turn(
                    "w_main", "conversation", "turn-a"
                )
                await self._collect(response)

            self.assertEqual(store.get("w_main", "conversation"), before)
            self.assertEqual(
                store.list_turn_events("w_main", "turn-a"),
                [{"sequence": 0, "kind": "final", "payload": {"text": "done"}}],
            )

    async def test_a_turn_of_another_conversation_is_not_readable_here(self) -> None:
        """Turn ids are global, so naming one is not permission to read it."""
        with TemporaryDirectory() as temporary_directory:
            store = self._store(Path(temporary_directory))
            store.create("w_main", "mine", "workspace-write")
            store.create("w_main", "theirs", "workspace-write")
            store.start_turn("w_main", "theirs", "turn-theirs")
            store.append_turn_event("w_main", "turn-theirs", "final", {"text": "x"})

            with mock.patch.object(app_module, "store", store):
                with self.assertRaises(HTTPException) as raised:
                    await app_module.replay_turn("w_main", "mine", "turn-theirs")

            self.assertEqual(raised.exception.status_code, 404)
