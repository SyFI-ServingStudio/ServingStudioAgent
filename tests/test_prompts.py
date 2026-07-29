from backend.codex_runtime.prompts import (
    _implementer_prompt,
    _orchestrator_prompt,
)


def test_initial_orchestrator_prompt_includes_selected_workspace_contract() -> None:
    prompt = _orchestrator_prompt(
        "Analyze the sweep.",
        is_resume=False,
        autonomous=True,
    )

    assert "active autonomous coordinator" in prompt
    assert "You are the orchestrator." in prompt
    assert "/workspace/AGENTS.md" not in prompt
    assert prompt.endswith("Newest user message:\nAnalyze the sweep.\n")


def test_initial_implementer_prompt_includes_default_workspace_contract() -> None:
    prompt = _implementer_prompt(
        "Change one file.",
        is_resume=False,
        autonomous=False,
    )

    assert "active human-in-the-loop coordinator" in prompt
    assert "You are the implementer." in prompt
    assert "/workspace/AGENTS.md" not in prompt
    assert prompt.endswith("Task:\nChange one file.\n")


def test_resumed_role_prompts_do_not_repeat_workspace_contract() -> None:
    assert _orchestrator_prompt(
        "Follow up.",
        is_resume=True,
        autonomous=True,
    ) == "Follow up."
    assert _implementer_prompt(
        "Continue.",
        is_resume=True,
        autonomous=True,
    ) == "Task:\nContinue.\n"
