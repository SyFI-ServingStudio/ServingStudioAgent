import json
import unittest
from pathlib import Path

from backend.agents_prompt import (
    AGENTS_PROMPT_MATRIX,
    ROLE_PROMPT_MATRIX,
    ensure_rendered,
    rendered_artifacts,
)
from backend.codex_runtime.prompts import (
    _assistant_prompt,
    _implementer_prompt,
    _orchestrator_handoff_prompt,
    _orchestrator_prompt,
    _orchestrator_repair_prompt,
    compose_failure_message,
    compose_final_message,
    parse_orchestrator,
    transport_failure_reason,
)
from backend.codex_runtime.codex_events import parse_commentary

PROMPT_DIRECTORY = Path(__file__).parents[1] / "backend" / "prompts"


class RenderedPromptTest(unittest.TestCase):
    """The mode-dependent files in `backend/prompts/` are gitignored build
    output, rendered from `backend/prompt_templates/` when
    `backend.codex_runtime.config` is imported.

    The runtime consumes them as real files (bind-mount source, container reuse
    `cmp -s`, fingerprint hashing), so the thing worth testing is no longer
    drift — it is that the import-time bootstrap actually put them on disk.
    """

    def test_importing_the_runtime_materializes_every_generated_prompt(self) -> None:
        # `backend.codex_runtime.prompts` is imported at module scope above,
        # which pulls in `config` and therefore runs `ensure_rendered()`.
        for filename, rendered in rendered_artifacts().items():
            with self.subTest(prompt=filename):
                path = PROMPT_DIRECTORY / filename
                self.assertTrue(path.is_file(), f"{filename} was never rendered")
                self.assertEqual(path.read_text("utf-8"), rendered)

    def test_a_hand_edited_artifact_is_overwritten_on_the_next_render(self) -> None:
        """Hand-edits to generated prompts are not a supported way to change the
        contract, so they must not survive — and a clean tree must not churn."""
        path = PROMPT_DIRECTORY / AGENTS_PROMPT_MATRIX["orchestrated", False]
        original = path.read_text("utf-8")
        self.addCleanup(path.write_text, original, encoding="utf-8")

        path.write_text(original + "\nHand-edited.\n", encoding="utf-8")
        self.assertIn(path.name, ensure_rendered())
        self.assertEqual(path.read_text("utf-8"), original)
        self.assertEqual(ensure_rendered(), [])

    def test_the_matrix_covers_both_modes_in_both_autonomy_settings(self) -> None:
        self.assertEqual(
            sorted(AGENTS_PROMPT_MATRIX),
            [
                ("orchestrated", False),
                ("orchestrated", True),
                ("single", False),
                ("single", True),
            ],
        )
        self.assertEqual(sorted(ROLE_PROMPT_MATRIX), ["orchestrated", "single"])


class RolePromptTest(unittest.TestCase):
    def test_orchestrator_schema_uses_provider_supported_subset(self) -> None:
        schema = json.loads(
            (PROMPT_DIRECTORY / "orchestrator.schema.json").read_text("utf-8")
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

    def test_assistant_schema_drops_delegate_and_its_task_field(self) -> None:
        schema = json.loads(
            (PROMPT_DIRECTORY / "assistant.schema.json").read_text("utf-8")
        )

        self.assertNotIn("oneOf", schema)
        self.assertEqual(
            schema["properties"]["action"]["enum"],
            ["progress", "milestone", "final_answer", "request_user_input"],
        )
        self.assertEqual(schema["required"], ["action", "message"])
        self.assertNotIn("task", schema["properties"])

    def test_agent_contracts_share_analyzer_selection_and_citation_rules(self) -> None:
        for (agent_mode, _), filename in AGENTS_PROMPT_MATRIX.items():
            prompt = (PROMPT_DIRECTORY / filename).read_text("utf-8")
            with self.subTest(prompt=filename):
                self.assertIn("operate-use-analyzer/SKILL.md", prompt)
                self.assertIn("/api/v1/sweeps?status=ready&limit=5", prompt)
                self.assertIn("Same-workspace results from other conversations", prompt)
                self.assertIn(
                    "sweep, run, timing prediction, kernel profile, or kernel",
                    prompt,
                )
                self.assertIn("`citations` map", prompt)
                self.assertIn("Copy the matching token unchanged", prompt)
                self.assertIn("Never assemble", prompt)
                self.assertIn('"action": "final_answer"', prompt)
                self.assertIn('"action": "progress"', prompt)
                self.assertIn('"action": "milestone"', prompt)
                self.assertIn('"action": "request_user_input"', prompt)
                # Only the orchestrated contract may offer a fifth action.
                # (`delegate` still appears in single-mode prose that rules it
                # out, so assert on the envelope example, not the bare word.)
                if agent_mode == "single":
                    self.assertNotIn('"action": "delegate"', prompt)
                    self.assertIn("no\n`delegate` action", prompt)
                else:
                    self.assertIn('"action": "delegate"', prompt)
                self.assertIn("reporting meaningful `progress` and `milestone`", prompt)
                self.assertIn(
                    "`tool call -> progress or milestone -> next tool",
                    prompt,
                )
                self.assertIn("Never concatenate envelopes", prompt)
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
        self.assertIn("user-visible intermediate output", prompt)
        self.assertIn("then immediately continue with the next genuine tool call", prompt)
        self.assertIn("Never\nconcatenate two envelopes", prompt)
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

    def test_initial_assistant_prompt_points_to_its_own_contract_section(self) -> None:
        prompt = _assistant_prompt(
            "Analyze the sweep.",
            is_resume=False,
            conversation_id="conversation-123",
        )

        self.assertIn("You are the VibeSim assistant.", prompt)
        self.assertIn("Read and follow `/workspace/AGENTS.md`", prompt)
        self.assertIn("Assistant Role", prompt)
        self.assertIn("/workspace/conversation-123_plan.md", prompt)
        self.assertIn("/workspace/conversation-123_progress.md", prompt)
        self.assertIn("there is no separate implementer to delegate to", prompt)
        self.assertNotIn("`task`", prompt.split("never emit a `task` field")[0])
        self.assertTrue(prompt.endswith("Newest user message:\nAnalyze the sweep.\n"))

    def test_resumed_assistant_repeats_the_workspace_contract(self) -> None:
        prompt = _assistant_prompt(
            "Follow up.",
            is_resume=True,
            conversation_id="conversation-123",
        )

        self.assertIn("You are the VibeSim assistant.", prompt)
        self.assertIn("Read and follow `/workspace/AGENTS.md`", prompt)
        self.assertTrue(prompt.endswith("Newest user message:\nFollow up.\n"))

    def test_single_mode_refuses_a_delegate_decision(self) -> None:
        """There is no implementer to hand to, so the envelope must be repaired
        rather than silently read as a non-terminal step."""
        delegation = '{"action":"delegate","message":"","task":"Change it."}'
        legacy = '{"action":"run_implementer","message":"","task":"Change it."}'

        for payload in (delegation, legacy):
            with self.subTest(payload=payload):
                self.assertIsNone(parse_orchestrator(payload, allow_delegate=False))
                self.assertIsNotNone(parse_orchestrator(payload))

        # Every other action still parses identically in single mode.
        self.assertEqual(
            parse_orchestrator(
                '{"action":"final_answer","message":"Done.","task":""}',
                allow_delegate=False,
            ),
            {"action": "final_answer", "message": "Done."},
        )

    def test_repair_prompt_preserves_unparsed_answer(self) -> None:
        prompt = _orchestrator_repair_prompt(
            "Analysis complete. Throughput rises with TP.",
            conversation_id="conversation-123",
        )

        self.assertIn("could not be parsed", prompt)
        self.assertIn("Do not redo completed analysis", prompt)
        self.assertIn("Analysis complete. Throughput rises with TP.", prompt)
        self.assertIn("/workspace/conversation-123_plan.md", prompt)


REPORTS = [
    "Implemented and committed the batched delivery trial. Commit fbed215.",
    "Investigated DecodeStream. 68082 token steps, zero mismatches.",
]


class FailureMessageTest(unittest.TestCase):
    """A failed turn must report the implementer's work by count. Quoting it
    attributes the implementer's first-person report to the assistant, which is
    what read as the two roles having been confused."""

    def test_summaries_are_counted_never_quoted(self) -> None:
        message = compose_failure_message("The gateway is down.", REPORTS)

        for report in REPORTS:
            self.assertNotIn(report, message)
        self.assertNotIn("### Implementer Summary", message)
        self.assertNotIn("**Round", message)
        self.assertIn("2 implementer rounds completed", message)
        self.assertIn("continue", message)

    def test_singular_round_reads_correctly(self) -> None:
        message = compose_failure_message("The gateway is down.", REPORTS[:1])

        self.assertIn("1 implementer round completed", message)

    def test_no_rounds_adds_nothing(self) -> None:
        self.assertEqual(
            compose_failure_message("The gateway is down.", []), "The gateway is down."
        )

    def test_it_matches_the_answer_path_contract(self) -> None:
        """`compose_final_message` already excludes the summaries; the failure
        path must not be the one place that reintroduces them."""
        self.assertEqual(
            compose_final_message("Done.", REPORTS),
            "Done.",
        )

    def test_reasons_name_the_transport_not_the_model(self) -> None:
        outage = transport_failure_reason(
            "orchestrator", {"code": "upstream_unavailable", "status": 503}
        )
        limited = transport_failure_reason(
            "implementer", {"code": "upstream_rate_limited", "status": 429}
        )
        stalled = transport_failure_reason(
            "orchestrator", {"code": "codex_call_timeout", "status": 0}
        )

        self.assertIn("503", outage)
        self.assertIn("not a model or parsing problem", outage)
        self.assertIn("429", limited)
        self.assertIn("rate limiting", limited)
        self.assertIn("idle timeout", stalled)
        for reason in (outage, limited, stalled):
            self.assertNotIn("parse the orchestrator decision", reason)


if __name__ == "__main__":
    unittest.main()
