from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.codex_runtime import turn as turn_module


class OrchestratorContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_continue_work_resumes_same_session_before_final(self) -> None:
        prompts: list[tuple[str, str | None]] = []

        async def fake_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        async def fake_run_codex(container, prompt, **kwargs):
            del container
            prompts.append((prompt, kwargs["session_id"]))
            if len(prompts) == 1:
                yield {
                    "kind": "session",
                    "role": "orchestrator",
                    "backend": "codexds",
                    "session_id": "session-1",
                }
                yield {
                    "kind": "final",
                    "text": (
                        '{"action":"continue_work",'
                        '"message":"Checking recovery state.","task":""}'
                    ),
                }
                return
            yield {
                "kind": "final",
                "text": (
                    '{"action":"user_message","message":"Completed answer.","task":""}'
                ),
            }

        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            with (
                patch.object(turn_module, "workspace_main_for", return_value=workspace),
                patch.object(turn_module, "prepare_workspace", return_value=workspace),
                patch.object(turn_module, "container_running", return_value=True),
                patch.object(turn_module, "ensure_container", return_value="container"),
                patch.object(turn_module, "write_managed_context"),
                patch.object(turn_module, "run_codex", fake_run_codex),
                patch.object(turn_module.asyncio, "to_thread", fake_to_thread),
            ):
                events = [
                    event
                    async for event in turn_module.run_turn(
                        "w_test",
                        "conversation-1",
                        "Analyze the sweep.",
                        sandbox="workspace-write",
                        orchestrator_backend="codexds",
                    )
                ]

        self.assertEqual(len(prompts), 2)
        self.assertIsNone(prompts[0][1])
        self.assertEqual(prompts[1][1], "session-1")
        self.assertIn("previous decision was `continue_work`", prompts[1][0])
        self.assertIn("Checking recovery state.", prompts[1][0])
        self.assertIn(
            {
                "kind": "intermediate_output",
                "role": "orchestrator",
                "backend": "codexds",
                "text": "Checking recovery state.",
            },
            events,
        )
        self.assertEqual(events[-1], {"kind": "final", "text": "Completed answer."})

    async def test_continue_work_has_a_finite_limit(self) -> None:
        calls = 0

        async def fake_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        async def fake_run_codex(container, prompt, **kwargs):
            nonlocal calls
            del container, prompt, kwargs
            calls += 1
            yield {
                "kind": "final",
                "text": (
                    '{"action":"continue_work","message":"Still checking.","task":""}'
                ),
            }

        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            with (
                patch.object(turn_module, "workspace_main_for", return_value=workspace),
                patch.object(turn_module, "prepare_workspace", return_value=workspace),
                patch.object(turn_module, "container_running", return_value=True),
                patch.object(turn_module, "ensure_container", return_value="container"),
                patch.object(turn_module, "write_managed_context"),
                patch.object(turn_module, "run_codex", fake_run_codex),
                patch.object(turn_module.asyncio, "to_thread", fake_to_thread),
            ):
                events = [
                    event
                    async for event in turn_module.run_turn(
                        "w_test",
                        "conversation-1",
                        "Analyze the sweep.",
                        sandbox="workspace-write",
                    )
                ]

        self.assertEqual(calls, turn_module.MAX_ORCHESTRATOR_CONTINUATIONS + 1)
        self.assertIn("repeatedly stopped", events[-1]["text"])

    async def test_unparsed_answer_is_repaired_in_same_session(self) -> None:
        prompts: list[tuple[str, str | None]] = []

        async def fake_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        async def fake_run_codex(container, prompt, **kwargs):
            del container
            prompts.append((prompt, kwargs["session_id"]))
            if len(prompts) == 1:
                yield {
                    "kind": "session",
                    "role": "orchestrator",
                    "backend": "codexds",
                    "session_id": "session-1",
                }
                yield {
                    "kind": "final",
                    "text": "Analysis complete. Throughput rises with TP.",
                }
                return
            yield {
                "kind": "final",
                "text": (
                    '{"action":"user_message",'
                    '"message":"Analysis complete. Throughput rises with TP.",'
                    '"task":""}'
                ),
            }

        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            with (
                patch.object(turn_module, "workspace_main_for", return_value=workspace),
                patch.object(turn_module, "prepare_workspace", return_value=workspace),
                patch.object(turn_module, "container_running", return_value=True),
                patch.object(turn_module, "ensure_container", return_value="container"),
                patch.object(turn_module, "write_managed_context"),
                patch.object(turn_module, "run_codex", fake_run_codex),
                patch.object(turn_module.asyncio, "to_thread", fake_to_thread),
            ):
                events = [
                    event
                    async for event in turn_module.run_turn(
                        "w_test",
                        "conversation-1",
                        "Analyze the sweep.",
                        sandbox="workspace-write",
                        orchestrator_backend="codexds",
                    )
                ]

        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[1][1], "session-1")
        self.assertIn("Do not redo completed analysis", prompts[1][0])
        self.assertIn("Analysis complete. Throughput rises with TP.", prompts[1][0])
        self.assertEqual(
            events[-1],
            {
                "kind": "final",
                "text": "Analysis complete. Throughput rises with TP.",
            },
        )


if __name__ == "__main__":
    unittest.main()
