import json
import unittest

from vibesim_agent.domain.decisions import parse_implementer, parse_orchestrator


class DecisionTests(unittest.TestCase):
    def test_non_string_action_is_invalid_instead_of_crashing(self):
        for action in ([], {}, 1, True):
            with self.subTest(action=action):
                text = json.dumps({"action": action, "message": "answer"})
                self.assertIsNone(parse_orchestrator(text))
                self.assertEqual(
                    parse_implementer(text, allow_reply_user=True),
                    {"action": "final_answer", "message": text},
                )

    def test_driver_terminal_and_nonterminal_actions(self):
        for action in ("final_answer", "request_user_input", "progress", "milestone"):
            with self.subTest(action=action):
                self.assertEqual(
                    parse_orchestrator(
                        json.dumps(
                            {
                                "action": action,
                                "message": "answer",
                                "task": "",
                            }
                        )
                    ),
                    {"action": action, "message": "answer"},
                )

    def test_single_mode_rejects_delegation_and_mixed_envelope(self):
        text = json.dumps({"action": "delegate", "task": "implement this"})
        self.assertIsNone(parse_orchestrator(text, allow_delegate=False))
        self.assertEqual(
            parse_orchestrator(text), {"action": "delegate", "task": "implement this"}
        )
        self.assertIsNone(
            parse_orchestrator(
                json.dumps(
                    {
                        "action": "delegate",
                        "task": "implement",
                        "message": "also answer",
                    }
                )
            )
        )

    def test_implementer_direct_reply_only_after_user_steer(self):
        text = json.dumps({"action": "reply_user", "message": "clarification"})
        self.assertEqual(
            parse_implementer(text, allow_reply_user=True),
            {"action": "reply_user", "message": "clarification"},
        )
        self.assertEqual(
            parse_implementer(text, allow_reply_user=False),
            {"action": "final_answer", "message": "clarification"},
        )
        self.assertEqual(
            parse_implementer("plain handoff", allow_reply_user=False),
            {"action": "final_answer", "message": "plain handoff"},
        )

    def test_legacy_alias_fence_and_escaped_newlines(self):
        self.assertEqual(
            parse_orchestrator(
                '```json\n{"action":"respond","message":"one\\\\ntwo"}\n```'
            ),
            {"action": "final_answer", "message": "one\ntwo"},
        )
        self.assertIsNone(parse_orchestrator("unstructured driver response"))
