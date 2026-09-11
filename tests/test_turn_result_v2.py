import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

from vibesim_agent.domain.results import project_turn_result
from vibesim_agent.domain.roles import AgentMode, Role, Sandbox
from vibesim_agent.domain.turns import Outcome, TurnInput, TurnResult


def rows(*events):
    return [
        {"kind": event["kind"], "payload": event, "sequence": index}
        for index, event in enumerate(events)
    ]


class TurnResultProjectionTests(unittest.TestCase):
    def setUp(self):
        self.request = TurnInput(
            "workspace",
            "conversation",
            "turn",
            "question",
            AgentMode.SINGLE,
            {},
            {Role.ASSISTANT: "previous-session"},
            "",
            sandbox=Sandbox.READ_ONLY,
            autonomous=True,
        )

    def test_collect_fields_match_legacy_and_terminal_result_is_authoritative(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures/legacy_turn_result.json").read_text()
        )
        events = rows(*fixture["events"])
        expected = fixture["collected"]
        expected.update(final="Normalized answer", outcome="final_answer", ok=True)
        before = copy.deepcopy(events)
        actual = project_turn_result(
            self.request, TurnResult(Outcome.ANSWER, "Normalized answer"), events
        )
        self.assertEqual(actual, expected)
        self.assertEqual(events, before)
        self.assertEqual(actual["sessions"], {"assistant": "latest"})

    def test_no_events_never_echoes_prior_sessions_and_keeps_empty_legacy_fields(self):
        output = project_turn_result(
            self.request, TurnResult(Outcome.INPUT, "Which model?"), []
        )
        self.assertEqual(output["sessions"], {})
        self.assertEqual(output["outcome"], "request_user_input")
        self.assertTrue(output["ok"])
        for key in (
            "tool_calls",
            "intermediate_outputs",
            "implementer_summaries",
            "delegated_tasks",
            "usages",
        ):
            self.assertEqual(output[key], [])
        self.assertNotIn("citations", output)
        self.assertNotIn("failure", output)
        self.assertNotIn("naming", output)

    def test_failure_error_field_uses_safe_message_and_never_reports_success(self):
        events = rows({"kind": "error", "text": "private host exception"})
        for metadata, code, message in (
            ({}, "", "The agent could not complete this turn."),
            ({"failure": {}}, "", "The agent could not complete this turn."),
            (
                {
                    "failure": {
                        "code": "runtime_timeout",
                        "message": "The turn timed out.",
                    }
                },
                "runtime_timeout",
                "The turn timed out.",
            ),
        ):
            with self.subTest(metadata=metadata):
                output = project_turn_result(
                    self.request,
                    TurnResult(Outcome.FAILED, "Failure explanation", metadata),
                    events,
                )
                self.assertFalse(output["ok"])
                self.assertIsNone(output["outcome"])
                self.assertEqual(output["final"], "Failure explanation")
                self.assertEqual(output["error"], message)
                self.assertEqual(output["failure_code"], code)

    def test_success_preserves_last_event_error_and_legacy_tool_collection(self):
        events = rows(
            {"kind": "error", "text": "first error"},
            {"kind": "tool_call", "text": "retry"},
            {"kind": "error", "text": "last error"},
        )
        output = project_turn_result(
            self.request, TurnResult(Outcome.ANSWER, "Answer"), events
        )
        self.assertFalse(output["ok"])
        self.assertEqual(output["error"], "last error")
        self.assertEqual(output["tool_calls"], ["first error", "retry", "last error"])
        self.assertEqual(output["outcome"], "final_answer")
        empty = project_turn_result(
            self.request, TurnResult(Outcome.ANSWER, "Answer"), rows({"kind": "error"})
        )
        self.assertFalse(empty["ok"])

    def test_outcome_and_nonempty_answer_are_both_required_for_ok(self):
        for outcome, text, metadata in (
            (Outcome.CANCELLED, "Stopped.", {}),
            (Outcome.FAILED, "Details", {}),
            (Outcome.ANSWER, "", {}),
            (Outcome.INPUT, " \n", {}),
            (Outcome.ANSWER, "Answer", {"failure": {}}),
        ):
            with self.subTest(outcome=outcome, text=text):
                output = project_turn_result(
                    self.request, TurnResult(outcome, text, metadata), []
                )
                self.assertFalse(output["ok"])
                self.assertEqual(
                    output["outcome"],
                    None if outcome is Outcome.FAILED else outcome.value,
                )

    def test_orchestrated_request_settings_do_not_depend_on_metadata(self):
        request = replace(
            self.request,
            mode=AgentMode.ORCHESTRATED,
            autonomous=False,
            sandbox=Sandbox.WORKSPACE_WRITE,
        )
        result = TurnResult(
            Outcome.ANSWER,
            "Answer",
            {
                "sessions": {"assistant": "spoof"},
                "delegated_tasks": ["spoof"],
                "citations": ["extra"],
            },
        )
        output = project_turn_result(request, result, [])
        self.assertEqual(
            (output["sandbox"], output["autonomous"], output["agent_mode"]),
            ("workspace-write", False, "orchestrated"),
        )
        self.assertEqual(output["sessions"], {})
        self.assertEqual(output["delegated_tasks"], [])
