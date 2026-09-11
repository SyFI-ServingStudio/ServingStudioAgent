from __future__ import annotations

import json
import logging
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.domain.roles import Role
from vibesim_agent.providers.base import AgentRequest, Model, Selection
from vibesim_agent.providers.codex.collector import CodexOutputCollector


def _collector() -> CodexOutputCollector:
    return CodexOutputCollector(
        AgentRequest(
            container="container",
            prompt="question",
            role=Role.ORCHESTRATOR,
            workspace_id="w_test",
            conversation_id="conversation-1",
            turn_id="turn-1",
            selection=Selection("deepseek", Model("deepseek-ai/DeepSeek-V4-Flash-0731", "DeepSeek", ("max",), "max"), "max", "default", "scope"),
        ), home=Path("/nonexistent-test-home"), idle_timeout=600, logger=logging.getLogger("collector-test"),
    )


def _completed_item(item: dict[str, object]) -> bytes:
    return json.dumps({"type": "item.completed", "item": item}).encode("utf-8")


class OutputCollectorTests(unittest.TestCase):
    def test_session_and_advisories_are_not_readiness_evidence(self):
        collector = _collector()
        collector.events_from_stdout_line(json.dumps(
            {"type": "thread.started", "thread_id": "session"}
        ).encode())
        collector.events_from_stdout_line(_error_event("Reconnecting..."))
        collector.events_from_stdout_line(_completed_item(
            {"type": "error", "message": "resuming with a different model"}
        ))
        self.assertFalse(collector.ready)
        collector.events_from_stdout_line(_completed_item(
            {"type": "command_execution", "command": "inspect"}
        ))
        self.assertTrue(collector.ready)

    def test_final_only_output_marks_ready_without_intermediate_event(self):
        collector = _collector()
        events = collector.events_from_stdout_line(_completed_item({
            "type": "agent_message", "text": "Done.", "phase": "final_answer"
        }))
        self.assertEqual(events, [])
        self.assertTrue(collector.ready)

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
    def test_bounded_stderr_preserves_early_status_but_success_suppresses_it(self):
        collector = _collector()
        collector.append_stderr_line(b"unexpected sta")
        collector.append_stderr_line(b"tus 429 Too Many Requests\n")
        for _ in range(100):
            collector.append_stderr_line(b"benign warning\n" * 100)
        self.assertLessEqual(len(collector.stderr_tail), 8192)
        self.assertGreater(collector.stderr_chars_seen, 8192)
        self.assertNotIn("429", collector.stderr_tail)
        self.assertEqual(collector.final_event(1)["failure"],
                         {"code": "upstream_rate_limited", "status": 429})
        collector.events_from_stdout_line(_completed_item(
            {"type": "agent_message", "text": "Recovered.", "phase": "final_answer"}
        ))
        self.assertEqual(collector.final_event(0), {"kind": "final", "text": "Recovered."})

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


def _tokens(input_tokens, cached_tokens, output_tokens):
    return {"type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": {"input_tokens": input_tokens,
                              "cached_input_tokens": cached_tokens,
                              "output_tokens": output_tokens}
    }}}


class SessionRecoveryTests(unittest.TestCase):
    def test_same_session_retains_primed_offset_and_token_baseline(self):
        with TemporaryDirectory() as directory:
            collector = _collector()
            collector.home = Path(directory)
            sessions = collector.home / "sessions"
            sessions.mkdir()
            path = sessions / "rollout-session-old.jsonl"
            path.write_text(json.dumps(_tokens(100, 30, 20)) + "\n", encoding="utf-8")
            collector.request = replace(collector.request, session_id="session-old")
            collector.current_session_id = "session-old"
            collector.prime_rollout_offset()
            offset = collector.rollout_offset
            collector.events_from_stdout_line(json.dumps(
                {"type": "thread.started", "thread_id": "session-old"}
            ).encode())
            self.assertEqual(collector.rollout_offset, offset)
            self.assertEqual(collector.rollout_file, path)
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(_tokens(150, 40, 35)) + "\n")
            self.assertEqual(collector.usage_event(10)["tokens"],
                             {"read": 10, "prefill": 40, "output": 15})

    def test_changed_session_discards_old_rollout_and_counts_new_session(self):
        with TemporaryDirectory() as directory:
            collector = _collector()
            collector.home = Path(directory)
            sessions = collector.home / "sessions"
            sessions.mkdir()
            old = sessions / "rollout-session-old.jsonl"
            old.write_text(json.dumps(_tokens(100, 30, 20)) + "\n", encoding="utf-8")
            new = sessions / "rollout-session-new.jsonl"
            new.write_text(json.dumps(_tokens(20, 5, 7)) + "\n" + json.dumps({
                "type": "event_msg", "payload": {"type": "task_complete",
                "last_agent_message": "New session answer."}
            }) + "\n", encoding="utf-8")
            collector.current_session_id = "session-old"
            collector.prime_rollout_offset()
            collector.rollout_terminal_text = "Old session answer."
            events = collector.events_from_stdout_line(json.dumps(
                {"type": "thread.started", "thread_id": "session-new"}
            ).encode())
            self.assertEqual(events[0]["session_id"], "session-new")
            self.assertIsNone(collector.rollout_file)
            self.assertIsNone(collector.tokens_baseline)
            self.assertIsNone(collector.rollout_terminal_text)
            self.assertEqual(collector.rollout_offset, 0)
            collector.poll_rollout_intermediate_outputs()
            self.assertTrue(collector.ready)
            self.assertEqual(collector.rollout_file, new)
            self.assertEqual(collector.final_event(0)["text"], "New session answer.")
            self.assertEqual(collector.usage_event(10)["tokens"],
                             {"read": 5, "prefill": 15, "output": 7})


if __name__ == "__main__":
    unittest.main()
