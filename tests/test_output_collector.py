from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def test_terminal_envelope_survives_trailing_todo_list(self) -> None:
        collector = _collector()
        final_text = '{"action":"final_answer","message":"Done.","task":""}'

        self.assertEqual(
            collector.events_from_stdout_line(
                _completed_item(
                    {
                        "type": "agent_message",
                        "text": final_text,
                        "phase": None,
                    }
                )
            ),
            [],
        )
        events = collector.events_from_stdout_line(
            _completed_item({"type": "todo_list"})
        )
        collector.append_stderr_line(
            b"worker quit with fatal: AuthRequired(Missing or invalid access token)\n"
        )

        self.assertEqual([event["kind"] for event in events], ["tool_call"])
        self.assertEqual(
            collector.final_event(0),
            {"kind": "final", "text": final_text},
        )

    def test_rollout_task_complete_recovers_a_flushed_free_form_final(self) -> None:
        collector = _collector()
        recovered_text = "Implemented and committed. 114 tests passed."
        collector.events_from_stdout_line(
            _completed_item(
                {
                    "type": "agent_message",
                    "text": recovered_text,
                    "phase": None,
                }
            )
        )
        flushed = collector.events_from_stdout_line(
            _completed_item({"type": "todo_list"})
        )
        self.assertEqual(flushed[0]["kind"], "intermediate_output")

        with TemporaryDirectory() as temporary_directory:
            rollout_file = Path(temporary_directory) / "rollout.jsonl"
            rollout_file.write_text(
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "task_complete",
                            "last_agent_message": recovered_text,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            collector.rollout_file = rollout_file
            collector.rollout_offset = 0
            collector.current_session_id = "session-1"
            self.assertEqual(collector.poll_rollout_intermediate_outputs(), [])

        self.assertEqual(
            collector.final_event(0),
            {"kind": "final", "text": recovered_text},
        )

    def test_rollout_final_answer_recovers_without_task_complete(self) -> None:
        collector = _collector()
        recovered_text = "Final handoff from the durable rollout."

        with TemporaryDirectory() as temporary_directory:
            rollout_file = Path(temporary_directory) / "rollout.jsonl"
            rollout_file.write_text(
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "phase": "final_answer",
                            "content": [
                                {"type": "output_text", "text": recovered_text}
                            ],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            collector.rollout_file = rollout_file
            collector.rollout_offset = 0
            collector.current_session_id = "session-1"
            self.assertEqual(collector.poll_rollout_intermediate_outputs(), [])

        self.assertEqual(
            collector.final_event(0),
            {"kind": "final", "text": recovered_text},
        )


GATEWAY_503 = (
    "unexpected status 503 Service Unavailable: Service temporarily "
    "unavailable, url: http://cayenne.cs.washington.edu:3456/responses, "
    "request id: de7c6e7b-c244-49b4-b2f1-6b45cbb644ee"
)


def _error_event(message: str) -> bytes:
    return json.dumps({"type": "error", "message": message}).encode("utf-8")


class TransportFailureTests(unittest.TestCase):
    def test_gateway_error_with_no_output_marks_the_call_failed(self) -> None:
        collector = _collector()
        events = collector.events_from_stdout_line(_error_event(GATEWAY_503))

        # The advisory still renders as the same warning line it always did.
        self.assertEqual(events[0]["kind"], "tool_call")

        final = collector.final_event(0)
        self.assertEqual(
            final["failure"], {"code": "upstream_unavailable", "status": 503}
        )
        # The gateway URL and request id must not reach the answer body.
        self.assertNotIn("cayenne", final["text"])
        self.assertNotIn("request id", final["text"])
        self.assertIn("503", final["text"])

    def test_a_reconnect_that_succeeded_is_not_a_failure(self) -> None:
        """The CLI logs the same status while retrying. A call that recovered
        and produced an answer must not be reported as failed."""
        collector = _collector()
        collector.events_from_stdout_line(
            _error_event(f"Reconnecting... 1/5 ({GATEWAY_503})")
        )
        collector.events_from_stdout_line(
            _completed_item(
                {
                    "type": "agent_message",
                    "text": '{"action":"final_answer","message":"Done.","task":""}',
                    "phase": None,
                }
            )
        )

        final = collector.final_event(0)
        self.assertNotIn("failure", final)
        self.assertEqual(
            final["text"], '{"action":"final_answer","message":"Done.","task":""}'
        )

    def test_stderr_only_gateway_failure_is_detected(self) -> None:
        collector = _collector()
        collector.append_stderr_line(
            b"exceeded retry limit, last status: 429 Too Many Requests, "
            b"request id: 01e49236\n"
        )

        final = collector.final_event(1)
        self.assertEqual(
            final["failure"], {"code": "upstream_rate_limited", "status": 429}
        )
        self.assertNotIn("request id", final["text"])

    def test_noisy_stderr_after_a_successful_call_is_not_a_failure(self) -> None:
        """A sandboxed command can print a status line of its own. Once the call
        produced an answer, stderr is never the cause."""
        collector = _collector()
        collector.events_from_stdout_line(
            _completed_item(
                {"type": "agent_message", "text": "All done.", "phase": None}
            )
        )
        collector.append_stderr_line(
            b"curl: the endpoint returned unexpected status 503\n"
        )

        self.assertNotIn("failure", collector.final_event(0))

    def test_idle_timeout_is_reported_as_a_failure(self) -> None:
        """An idle timeout also ends with no decision to parse, so it must not
        be sent through the repair loop either."""
        collector = _collector()
        collector.mark_timed_out()
        collector.timeout_event()

        final = collector.final_event(None)
        self.assertEqual(final["failure"], {"code": "codex_call_timeout", "status": 0})
        self.assertIn("idle timeout", final["text"])


if __name__ == "__main__":
    unittest.main()
