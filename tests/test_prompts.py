import json
import unittest
from pathlib import Path

from backend.codex_runtime.prompts import (
    _implementer_prompt,
    _orchestrator_handoff_prompt,
    _orchestrator_prompt,
    _orchestrator_repair_prompt,
    parse_orchestrator,
)
from backend.codex_runtime.codex_events import parse_commentary


class RolePromptTest(unittest.TestCase):
    def test_orchestrator_schema_uses_provider_supported_subset(self) -> None:
        prompt_directory = Path(__file__).parents[1] / "backend" / "prompts"
        schema = json.loads(
            (prompt_directory / "orchestrator.schema.json").read_text("utf-8")
        )

        self.assertNotIn("oneOf", schema)
        self.assertEqual(
            schema["properties"]["action"]["enum"],
            [
                "progress",
                "milestone",
                "final_answer",
                "request_user_input",
                "delegate",
            ],
        )
        self.assertEqual(schema["required"], ["action", "message", "task"])

    def test_agent_contracts_share_analyzer_selection_and_citation_rules(self) -> None:
        prompt_directory = Path(__file__).parents[1] / "backend" / "prompts"
        prompts = [
            (prompt_directory / filename).read_text("utf-8")
            for filename in ("AGENTS.md", "AGENTS.autonomous.md")
        ]

        for prompt in prompts:
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("operate-use-analyzer/SKILL.md", prompt)
                self.assertIn("/api/v1/sweeps?status=ready&limit=5", prompt)
                self.assertIn("Same-workspace results from other conversations", prompt)
                self.assertIn("adjacent complete citation token", prompt)
                self.assertIn("Never assemble", prompt)
                self.assertIn('"action": "final_answer"', prompt)
                self.assertIn('"action": "progress"', prompt)
                self.assertIn('"action": "milestone"', prompt)
                self.assertIn('"action": "request_user_input"', prompt)
                self.assertIn('"action": "delegate"', prompt)
                self.assertNotIn("continue_work", prompt)

    def test_initial_orchestrator_prompt_points_to_workspace_contract(self) -> None:
        prompt = _orchestrator_prompt(
            "Analyze the sweep.",
            is_resume=False,
            conversation_id="conversation-123",
        )

        self.assertIn("You are the orchestrator.", prompt)
        self.assertIn("Read and follow `/workspace/AGENTS.md`", prompt)
        self.assertIn("`conversation-123`", prompt)
        self.assertIn("/workspace/conversation-123_plan.md", prompt)
        self.assertIn("/workspace/conversation-123_progress.md", prompt)
        self.assertTrue(prompt.endswith("Newest user message:\nAnalyze the sweep.\n"))

    def test_initial_implementer_prompt_points_to_workspace_contract(self) -> None:
        prompt = _implementer_prompt(
            "Change one file.",
            is_resume=False,
        )

        self.assertTrue(prompt.startswith("You are implementor."))
        self.assertIn("Read and follow `/workspace/AGENTS.md`", prompt)
        self.assertTrue(prompt.endswith("Task:\nChange one file.\n"))

    def test_resumed_roles_repeat_workspace_contract(self) -> None:
        orchestrator_prompt = _orchestrator_prompt(
            "Follow up.",
            is_resume=True,
            conversation_id="conversation-123",
        )
        self.assertIn("You are the orchestrator.", orchestrator_prompt)
        self.assertIn("Read and follow `/workspace/AGENTS.md`", orchestrator_prompt)
        self.assertTrue(
            orchestrator_prompt.endswith("Newest user message:\nFollow up.\n")
        )
        implementer_prompt = _implementer_prompt(
            "Continue.",
            is_resume=True,
        )
        self.assertEqual(
            implementer_prompt,
            "You are implementor. Read and follow `/workspace/AGENTS.md`, especially "
            "the Implementer Role section.\n\nTask:\nContinue.\n",
        )
        self.assertIn("Read and follow `/workspace/AGENTS.md`", implementer_prompt)

    def test_implementer_handoff_reasserts_orchestrator_contract(self) -> None:
        prompt = _orchestrator_handoff_prompt(
            "Change one file.",
            "Implemented and tested.",
            conversation_id="conversation-123",
        )

        self.assertTrue(prompt.startswith("You are the orchestrator."))
        self.assertIn("Read and follow `/workspace/AGENTS.md`", prompt)
        self.assertIn("/workspace/conversation-123_plan.md", prompt)
        self.assertIn("Delegated task:\nChange one file.", prompt)
        self.assertIn("Implementer summary:\nImplemented and tested.", prompt)

    def test_parse_terminal_decisions_distinguishes_answer_and_user_input(self) -> None:
        answer = parse_orchestrator(
            '{"action":"final_answer","message":"Completed.","task":""}'
        )
        question = parse_orchestrator(
            '{"action":"request_user_input","message":"Which GPU?","task":""}'
        )

        self.assertEqual(
            answer,
            {"action": "final_answer", "message": "Completed."},
        )
        self.assertEqual(
            question,
            {"action": "request_user_input", "message": "Which GPU?"},
        )

    def test_parse_legacy_actions_normalizes_without_reexposing_them(self) -> None:
        answer = parse_orchestrator(
            '{"action":"user_message","message":"Completed.","task":""}'
        )
        delegation = parse_orchestrator(
            '{"action":"run_implementer","message":"","task":"Change it."}'
        )

        self.assertEqual(
            answer,
            {"action": "final_answer", "message": "Completed."},
        )
        self.assertEqual(
            delegation,
            {"action": "delegate", "task": "Change it."},
        )

    def test_progress_and_milestone_parse_as_non_terminal_decisions(self) -> None:
        self.assertEqual(
            parse_orchestrator(
                '{"action":"progress","message":"Still checking.","task":""}'
            ),
            {"action": "progress", "message": "Still checking."},
        )
        self.assertEqual(
            parse_orchestrator(
                '{"action":"milestone","message":"Sweep ready.","task":""}'
            ),
            {"action": "milestone", "message": "Sweep ready."},
        )

    def test_commentary_envelopes_preserve_progress_level(self) -> None:
        self.assertEqual(
            parse_commentary(
                '{"action":"progress","message":"Checking.","task":""}'
            ),
            ("Checking.", "progress"),
        )
        self.assertEqual(
            parse_commentary(
                '{"action":"milestone","message":"Validated.","task":""}'
            ),
            ("Validated.", "milestone"),
        )
        self.assertEqual(parse_commentary("Legacy prose."), ("Legacy prose.", "progress"))

    def test_parse_rejects_invalid_action_field_combinations(self) -> None:
        invalid_decisions = [
            '{"action":"final_answer","message":"","task":""}',
            '{"action":"final_answer","message":"Done.","task":"More work"}',
            '{"action":"request_user_input","message":"","task":""}',
            '{"action":"delegate","message":"Starting.","task":"Change it."}',
            '{"action":"delegate","message":"","task":""}',
        ]

        for decision in invalid_decisions:
            with self.subTest(decision=decision):
                self.assertIsNone(parse_orchestrator(decision))

    def test_repair_prompt_preserves_unparsed_answer(self) -> None:
        prompt = _orchestrator_repair_prompt(
            "Analysis complete. Throughput rises with TP.",
            conversation_id="conversation-123",
        )

        self.assertIn("could not be parsed", prompt)
        self.assertIn("Do not redo completed analysis", prompt)
        self.assertIn("Analysis complete. Throughput rises with TP.", prompt)
        self.assertIn("/workspace/conversation-123_plan.md", prompt)


if __name__ == "__main__":
    unittest.main()
