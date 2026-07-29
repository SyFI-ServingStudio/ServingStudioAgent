import unittest

from backend.codex_runtime.prompts import (
    _implementer_prompt,
    _orchestrator_prompt,
)


class RolePromptTest(unittest.TestCase):
    def test_initial_orchestrator_prompt_points_to_workspace_contract(self) -> None:
        prompt = _orchestrator_prompt(
            "Analyze the sweep.",
            is_resume=False,
        )

        self.assertIn("You are the orchestrator.", prompt)
        self.assertIn("Read and follow `/workspace/AGENTS.md`", prompt)
        self.assertTrue(prompt.endswith("Newest user message:\nAnalyze the sweep.\n"))

    def test_initial_implementer_prompt_points_to_workspace_contract(self) -> None:
        prompt = _implementer_prompt(
            "Change one file.",
            is_resume=False,
        )

        self.assertIn("You are the implementer.", prompt)
        self.assertIn("Read and follow `/workspace/AGENTS.md`", prompt)
        self.assertTrue(prompt.endswith("Task:\nChange one file.\n"))

    def test_resumed_role_prompts_do_not_repeat_workspace_contract(self) -> None:
        self.assertEqual(
            _orchestrator_prompt(
                "Follow up.",
                is_resume=True,
            ),
            "Follow up.",
        )
        self.assertEqual(
            _implementer_prompt(
                "Continue.",
                is_resume=True,
            ),
            "Task:\nContinue.\n",
        )


if __name__ == "__main__":
    unittest.main()
