import unittest

from backend.codex_runtime.prompts import (
    _implementer_prompt,
    _orchestrator_handoff_prompt,
    _orchestrator_prompt,
)


class RolePromptTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
