import json
import unittest
from dataclasses import replace
from pathlib import Path

from vibesim_agent.domain.roles import Role
from vibesim_agent.providers.base import AgentRequest, Model, OutputMode, Selection
from vibesim_agent.providers.claude.events import ClaudeOutputCollector


class ClaudeEventsTests(unittest.TestCase):
    def collector(self, *, structured=True):
        model = Model(
            "claude",
            "Claude",
            ("high",),
            "high",
            output_mode=OutputMode.STRUCTURED if structured else OutputMode.PROMPT,
        )
        request = AgentRequest(
            "w",
            "c",
            "t",
            Role.ASSISTANT,
            "question",
            "container",
            Selection("provider", model, "high", "default", "scope"),
            session_id="owner",
            output_schema=Path("/contracts/assistant.json"),
        )
        schema = json.loads(
            (
                Path(__file__).parents[1]
                / "vibesim_agent/prompts/contracts/assistant.schema.json"
            ).read_text()
        )
        return ClaudeOutputCollector(request, schema=schema)

    def consume(self, collector, event):
        return collector.events_from_stdout_line(json.dumps(event).encode())

    def result(self, **overrides):
        return {
            "type": "result",
            "subtype": "success",
            "session_id": "owner",
            "structured_output": {"action": "final_answer", "message": "Done"},
            "result": "raw fallback",
            **overrides,
        }

    def test_init_and_malformed_output_do_not_make_failed_call_resumable(self):
        collector = self.collector()
        events = self.consume(
            collector, {"type": "system", "subtype": "init", "session_id": "new"}
        )
        self.assertEqual([e["kind"] for e in events], ["session"])
        for value in (
            None,
            [],
            {"type": "assistant", "message": 42},
            {"type": "assistant", "message": {"content": []}},
            {"type": "user"},
            {"type": "stream_event"},
        ):
            self.assertEqual(self.consume(collector, value), [])
        self.assertEqual(collector.events_from_stdout_line(b"broken json"), [])
        self.assertFalse(collector.ready)
        finished = collector.finish(1, 10)
        self.assertEqual([e["kind"] for e in finished], ["usage", "final"])
        self.assertEqual(finished[-1]["failure"]["code"], "agent_runtime_failure")

    def test_assistant_progress_tools_deduplicate_without_routing(self):
        collector = self.collector()
        event = {
            "type": "assistant",
            "uuid": "message",
            "message": {
                "content": [
                    {"type": "text", "text": '{"action":"delegate","task":"wrong"}'},
                    {
                        "type": "text",
                        "text": '{"action":"progress","message":"Reading"}',
                    },
                    {"type": "tool_use", "name": "Read"},
                ]
            },
        }
        events = self.consume(collector, event)
        self.assertEqual(
            [e["kind"] for e in events],
            ["role_ready", "intermediate_output", "tool_call"],
        )
        self.assertEqual(events[1]["text"], "Reading")
        self.assertIsNone(collector.result)
        self.assertEqual(self.consume(collector, event), [])
        event["uuid"] = {}
        self.consume(collector, event)

    def test_tool_call_names_what_the_step_runs(self):
        collector = self.collector()
        long_command = "uv run python -m launcher " + "x" * 200
        event = {
            "type": "assistant",
            "uuid": "tools",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {
                            "command": "git status --short",
                            "description": "Show  working\ntree status",
                        },
                    },
                    {"type": "tool_use", "name": "Bash", "input": {"command": long_command}},
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "/w/a.py"}},
                    {"type": "tool_use", "name": "TodoWrite", "input": {"todos": []}},
                ]
            },
        }
        texts = [e["text"] for e in self.consume(collector, event) if e["kind"] == "tool_call"]
        self.assertEqual(texts[0], "assistant: Bash — Show working tree status")
        self.assertTrue(texts[1].startswith("assistant: Bash — uv run python -m launcher x"))
        self.assertTrue(texts[1].endswith("…"))
        self.assertEqual(len(texts[1]), len("assistant: Bash — ") + 100)
        self.assertEqual(texts[2], "assistant: Read — /w/a.py")
        self.assertEqual(texts[3], "assistant: TodoWrite")

    def test_nested_session_and_result_never_replace_owner(self):
        collector = self.collector()
        self.assertEqual(
            self.consume(
                collector, self.result(parent_tool_use_id="nested", session_id="other")
            ),
            [],
        )
        self.assertEqual(collector.session_id, "owner")
        self.assertIsNone(collector.result)
        self.assertFalse(collector.ready)

    def test_malformed_action_does_not_prevent_following_final_result(self):
        for action in ([], {}, None, 42):
            with self.subTest(action=action):
                collector = self.collector()
                events = self.consume(
                    collector,
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {"action": action, "message": "bad"}
                                    ),
                                },
                            ]
                        },
                    },
                )
                self.assertEqual([event["kind"] for event in events], ["role_ready"])
                self.assertIsNone(collector.result)
                self.consume(collector, self.result())
                final = collector.finish(0, 10)[-1]
                self.assertNotIn("failure", final)
                self.assertEqual(json.loads(final["text"])["message"], "Done")

    def test_result_schema_and_exit_status_control_final(self):
        for payload, exitcode, timeout, code in [
            (self.result(), 0, False, None),
            (self.result(is_error=True), 0, False, "agent_runtime_failure"),
            (self.result(), 1, False, "agent_runtime_failure"),
            (self.result(), 0, True, "agent_call_timeout"),
            (self.result(structured_output=None), 0, False, "agent_invalid_output"),
            (
                self.result(structured_output={"action": "delegate", "task": "bad"}),
                0,
                False,
                "agent_invalid_output",
            ),
        ]:
            with self.subTest(code=code, payload=payload):
                collector = self.collector()
                self.consume(collector, payload)
                final = collector.finish(exitcode, 10, timed_out=timeout)[-1]
                if code:
                    self.assertEqual(final["failure"]["code"], code)
                    self.assertEqual(final["text"], "")
                else:
                    self.assertEqual(json.loads(final["text"])["message"], "Done")

    def test_usage_and_prompt_only_capability(self):
        collector = self.collector(structured=False)
        self.consume(
            collector,
            self.result(
                usage={
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 30,
                    "output_tokens": 40,
                }
            ),
        )
        usage, final = collector.finish(0, 100)
        self.assertEqual(usage["tokens"], {"read": 30, "prefill": 30, "output": 40})
        self.assertEqual(final["text"], "raw fallback")
        for value in (float("nan"), float("inf"), True, -1, "secret"):
            self.consume(collector, self.result(usage={"output_tokens": value}))
            self.assertEqual(collector.finish(0, 1)[0]["tokens"]["output"], 0)

    def test_structured_request_requires_explicit_schema(self):
        request = self.collector().request
        with self.assertRaisesRegex(ValueError, "requires a schema"):
            ClaudeOutputCollector(request)
        ClaudeOutputCollector(replace(request, output_schema=None))
