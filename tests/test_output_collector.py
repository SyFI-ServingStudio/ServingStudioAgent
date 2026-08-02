from __future__ import annotations

import json
import unittest

from backend.codex_runtime.exec_types import CodexExecRequest
from backend.codex_runtime.output_collector import CodexOutputCollector


def _collector() -> CodexOutputCollector:
    return CodexOutputCollector(
        CodexExecRequest(
            container="container",
            prompt="question",
            label="orchestrator",
            workspace_id="w_test",
            conversation_id="conversation-1",
            turn_id="turn-1",
            model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
            effort="max",
        )
    )


def _completed_item(item: dict[str, object]) -> bytes:
    return json.dumps({"type": "item.completed", "item": item}).encode("utf-8")


class OutputCollectorTests(unittest.TestCase):
    def test_unphased_text_becomes_progress_when_a_tool_follows(self) -> None:
        collector = _collector()

        self.assertEqual(
            collector.events_from_stdout_line(
                _completed_item(
                    {
                        "type": "agent_message",
                        "text": "I will prepare the sweep, then run it.",
                        "phase": None,
                    }
                )
            ),
            [],
        )
        events = collector.events_from_stdout_line(
            _completed_item(
                {
                    "type": "command_execution",
                    "command": "uv run python -m launcher preset.yaml",
                }
            )
        )

        self.assertEqual(events[0]["kind"], "intermediate_output")
        self.assertEqual(events[0]["level"], "progress")
        self.assertEqual(events[0]["text"], "I will prepare the sweep, then run it.")
        self.assertEqual(events[1]["kind"], "tool_call")

    def test_unphased_milestone_envelope_keeps_its_level(self) -> None:
        collector = _collector()
        collector.events_from_stdout_line(
            _completed_item(
                {
                    "type": "agent_message",
                    "text": (
                        '{"action":"milestone","message":"Sweep ready.",'
                        '"task":""}'
                    ),
                    "phase": None,
                }
            )
        )

        events = collector.events_from_stdout_line(
            _completed_item({"type": "command_execution", "command": "inspect"})
        )

        self.assertEqual(events[0]["kind"], "intermediate_output")
        self.assertEqual(events[0]["level"], "milestone")
        self.assertEqual(events[0]["text"], "Sweep ready.")

    def test_last_unphased_text_remains_the_final_output(self) -> None:
        collector = _collector()
        collector.events_from_stdout_line(
            _completed_item(
                {
                    "type": "agent_message",
                    "text": '{"action":"final_answer","message":"Done.","task":""}',
                    "phase": None,
                }
            )
        )

        self.assertEqual(
            collector.final_event(0),
            {
                "kind": "final",
                "text": '{"action":"final_answer","message":"Done.","task":""}',
            },
        )


if __name__ == "__main__":
    unittest.main()
