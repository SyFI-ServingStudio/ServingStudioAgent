from __future__ import annotations

import unittest

from backend.codex_runtime.codex_events import _translate, transport_failure

# Verbatim from a real outage: the gateway at cayenne:3456 returned 503 for
# 12+ hours, and the runtime reported it as a decision-parsing failure.
RECONNECT_503 = (
    "Reconnecting... 1/5 (unexpected status 503 Service Unavailable: Service "
    "temporarily unavailable, url: http://cayenne.cs.washington.edu:3456/"
    "responses, request id: 21d5811d-e639-426d-adae-60f5a31367f8)"
)
TERMINAL_503 = (
    "unexpected status 503 Service Unavailable: Service temporarily "
    "unavailable, url: http://cayenne.cs.washington.edu:3456/responses, "
    "request id: de7c6e7b-c244-49b4-b2f1-6b45cbb644ee"
)
RETRY_LIMIT_429 = (
    "exceeded retry limit, last status: 429 Too Many Requests, request id: "
    "01e49236-fd81-46da-8b61-a5748dab6bbd"
)


class TransportFailureTests(unittest.TestCase):
    def test_gateway_outage_and_rate_limit_are_classified(self) -> None:
        self.assertEqual(
            transport_failure(TERMINAL_503),
            {"code": "upstream_unavailable", "status": 503},
        )
        self.assertEqual(
            transport_failure(RECONNECT_503),
            {"code": "upstream_unavailable", "status": 503},
        )
        self.assertEqual(
            transport_failure(RETRY_LIMIT_429),
            {"code": "upstream_rate_limited", "status": 429},
        )

    def test_advisories_and_unretryable_statuses_are_not_failures(self) -> None:
        # Codex reports a within-family model switch as an error item; it is
        # non-fatal and carries no status token.
        self.assertIsNone(
            transport_failure(
                "recorded with model gpt-5.6-sol but resuming with gpt-5.6-terra"
            )
        )
        # A bad or unauthorized request will not succeed on retry, so "the
        # service is down, continue later" would be the wrong advice.
        self.assertIsNone(transport_failure("unexpected status 400 Bad Request"))
        self.assertIsNone(transport_failure("unexpected status 401 Unauthorized"))

    def test_a_status_number_in_prose_is_not_a_failure(self) -> None:
        """Model output routinely contains bare numbers; only the CLI's own
        status phrasing counts."""
        self.assertIsNone(
            transport_failure("the benchmark completed 503 requests in 5.03 ms")
        )
        self.assertIsNone(transport_failure("429"))


class TranslateTests(unittest.TestCase):
    def test_error_event_keeps_its_warning_line_and_carries_the_failure(self) -> None:
        events = _translate({"type": "error", "message": TERMINAL_503})

        self.assertEqual(len(events), 1)
        # The advisory line is unchanged, so reconnect notices keep rendering
        # exactly as they do today.
        self.assertEqual(events[0]["kind"], "tool_call")
        self.assertEqual(events[0]["text"], f"warning: {TERMINAL_503}")
        self.assertEqual(
            events[0]["transport_failure"],
            {"code": "upstream_unavailable", "status": 503},
        )

    def test_ordinary_error_event_carries_no_failure(self) -> None:
        events = _translate({"type": "error", "message": "something odd happened"})

        self.assertEqual(
            events, [{"kind": "tool_call", "text": "warning: something odd happened"}]
        )

    def test_completed_items_are_unchanged(self) -> None:
        events = _translate(
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "ls -la"},
            }
        )

        self.assertEqual(events, [{"kind": "tool_call", "text": "$ ls -la"}])


if __name__ == "__main__":
    unittest.main()
