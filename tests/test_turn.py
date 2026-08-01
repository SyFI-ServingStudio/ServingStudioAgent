from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.codex_runtime import turn as turn_module


class OrchestratorDecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_progress_checkpoint_resumes_same_session(self) -> None:
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
                    "model": "gpt-5.6-terra",
                    "effort": "high",
                    "session_id": "session-1",
                }
                yield {
                    "kind": "final",
                    "text": (
                        '{"action":"milestone","message":"Sweep ready.",'
                        '"task":""}'
                    ),
                }
                return
            yield {
                "kind": "final",
                "text": (
                    '{"action":"final_answer","message":"TP4 wins.",'
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
                        orchestrator_runtime={
                            "model": "gpt-5.6-terra",
                            "effort": "high",
                        },
                    )
                ]

        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[1][1], "session-1")
        self.assertIn("previous call ended with a non-terminal `milestone`", prompts[1][0])
        self.assertIn(
            {
                "kind": "intermediate_output",
                "role": "orchestrator",
                "model": "gpt-5.6-terra",
                "effort": "high",
                "level": "milestone",
                "text": "Sweep ready.",
            },
            events,
        )
        self.assertEqual(
            events[-1],
            {"kind": "final", "outcome": "final_answer", "text": "TP4 wins."},
        )

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
                    "model": "gpt-5.6-terra",
                    "effort": "high",
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
                    '{"action":"final_answer",'
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
                        orchestrator_runtime={
                            "model": "gpt-5.6-terra",
                            "effort": "high",
                        },
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
                "outcome": "final_answer",
                "text": "Analysis complete. Throughput rises with TP.",
            },
        )


if __name__ == "__main__":
    unittest.main()
