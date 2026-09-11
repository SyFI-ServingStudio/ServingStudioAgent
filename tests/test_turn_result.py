"""How a turn's `final` event is read — the one place, for every entry point.

The JSON endpoints and the browser stream both act on this event, and they used
to interpret it separately. The failure mode of two copies is not a crash: it is
one caller quietly calling an ending an answer while the other does not, so the
same turn reads differently depending on which door it came through.
"""

from __future__ import annotations

import unittest

from backend.turn_result import collect_turn_event, new_turn_result, read_final_event


def _result() -> dict:
    return new_turn_result(
        conversation_id="c1", turn_id="t", sandbox="workspace-write", autonomous=False
    )


class ReadFinalEventTests(unittest.TestCase):
    def test_reports_the_outcome_a_role_named(self) -> None:
        self.assertEqual(
            read_final_event({"outcome": "request_user_input"}),
            ("request_user_input", ""),
        )

    def test_a_final_that_says_nothing_is_an_answer(self) -> None:
        # A role that produced a final and did not say why is answering; that is
        # the whole of what a role's own `final` means.
        self.assertEqual(read_final_event({"text": "done"}), ("final_answer", ""))

    def test_an_unknown_outcome_is_not_carried_through(self) -> None:
        # Whatever a newer runtime may put here, this build can only act on the
        # outcomes it knows. Passing an unrecognised one along would put a word
        # in the store that every reader downstream then has to guess at.
        self.assertEqual(read_final_event({"outcome": "gave_up"}), ("final_answer", ""))

    def test_a_failure_leaves_no_outcome_to_report(self) -> None:
        # The two are exclusive: a turn that failed never reached a decision.
        self.assertEqual(
            read_final_event({"failure": {"code": "agent_call_timeout"}}),
            (None, "agent_call_timeout"),
        )

    def test_a_failure_with_no_code_is_still_a_failure(self) -> None:
        # Asserted through a consumer, because the tuple alone cannot show the
        # bug this guards: every caller tests the code for truth, so an empty
        # one would leave the turn neither failed nor answered — and it would
        # then reach the reader as a blank answer.
        result = _result()
        collect_turn_event(result, {"kind": "final", "text": "", "failure": {}})
        self.assertIsNone(result["outcome"])
        self.assertTrue(result["failure_code"])


class CollectTurnEventTests(unittest.TestCase):
    """That the JSON path reads the event through the same rule."""

    def test_carries_the_outcome_into_the_result(self) -> None:
        result = _result()
        collect_turn_event(
            result, {"kind": "final", "text": "here", "outcome": "request_user_input"}
        )
        self.assertEqual(result["outcome"], "request_user_input")
        self.assertEqual(result["final"], "here")
        self.assertEqual(result["failure_code"], "")

    def test_a_failed_turn_reports_its_code_and_no_outcome(self) -> None:
        result = _result()
        collect_turn_event(
            result,
            {"kind": "final", "text": "gave up", "failure": {"code": "agent_call_timeout"}},
        )
        self.assertIsNone(result["outcome"])
        self.assertEqual(result["failure_code"], "agent_call_timeout")


if __name__ == "__main__":
    unittest.main()
