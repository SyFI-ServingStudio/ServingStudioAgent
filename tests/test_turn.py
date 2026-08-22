from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.codex_runtime import turn as turn_module
from backend.codex_runtime.config import (
    ASSISTANT_SCHEMA_IN_CONTAINER,
    CODEXDS_MODEL,
    ORCHESTRATOR_SCHEMA_IN_CONTAINER,
    driving_role_for_agent_mode,
)


class OrchestratorDecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_deepseek_omits_schema_while_gpt_keeps_it(self) -> None:
        async def run_for(
            model_id: str, agent_mode: str = "orchestrated"
        ) -> str | None:
            captured_schema: str | None = None

            async def fake_to_thread(function, *args, **kwargs):
                return function(*args, **kwargs)

            async def fake_run_codex(container, prompt, **kwargs):
                nonlocal captured_schema
                del container, prompt
                captured_schema = kwargs["output_schema"]
                yield {
                    "kind": "final",
                    "text": (
                        '{"action":"final_answer","message":"Done.",'
                        '"task":""}'
                    ),
                }

            with TemporaryDirectory() as temporary_directory:
                workspace = Path(temporary_directory)
                with (
                    patch.object(
                        turn_module, "workspace_main_for", return_value=workspace
                    ),
                    patch.object(
                        turn_module, "prepare_workspace", return_value=workspace
                    ),
                    patch.object(turn_module, "container_running", return_value=True),
                    patch.object(
                        turn_module, "ensure_container", return_value="container"
                    ),
                    patch.object(turn_module, "write_managed_context"),
                    patch.object(turn_module, "run_codex", fake_run_codex),
                    patch.object(turn_module.asyncio, "to_thread", fake_to_thread),
                ):
                    events = [
                        event
                        async for event in turn_module.run_turn(
                            "w_test",
                            "conversation-1",
                            "Analyze the sweep.",
                            sandbox="workspace-write",
                            agent_mode=agent_mode,
                            role_runtimes={
                                driving_role_for_agent_mode(agent_mode): {
                                    "model": model_id,
                                    "effort": (
                                        "max" if model_id == CODEXDS_MODEL else "high"
                                    ),
                                }
                            },
                        )
                    ]
            self.assertEqual(events[-1]["text"], "Done.")
            return captured_schema

        deepseek_schema = await run_for(CODEXDS_MODEL)
        gpt_schema = await run_for("gpt-5.6-terra")
        single_schema = await run_for("gpt-5.6-terra", agent_mode="single")

        self.assertIsNone(deepseek_schema)
        self.assertEqual(gpt_schema, ORCHESTRATOR_SCHEMA_IN_CONTAINER)
        self.assertEqual(single_schema, ASSISTANT_SCHEMA_IN_CONTAINER)

    async def test_final_progress_checkpoint_resumes_same_session(self) -> None:
        prompts: list[tuple[str, str | None]] = []

        async def fake_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        async def fake_run_codex(container, prompt, **kwargs):
            del container
            prompts.append((prompt, kwargs["session_id"]))
            if len(prompts) == 1:
                yield {
                    "kind": "session",
                    "role": "orchestrator",
                    "model": "gpt-5.6-terra",
                    "effort": "high",
                    "session_id": "session-1",
                }
                yield {
                    "kind": "final",
                    "text": (
                        '{"action":"milestone","message":"Sweep ready.",'
                        '"task":""}'
                    ),
                }
                return
            yield {
                "kind": "final",
                "text": (
                    '{"action":"final_answer","message":"TP4 wins.",'
                    '"task":""}'
                ),
            }

        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            with (
                patch.object(turn_module, "workspace_main_for", return_value=workspace),
                patch.object(turn_module, "prepare_workspace", return_value=workspace),
                patch.object(turn_module, "container_running", return_value=True),
                patch.object(turn_module, "ensure_container", return_value="container"),
                patch.object(turn_module, "write_managed_context"),
                patch.object(turn_module, "run_codex", fake_run_codex),
                patch.object(turn_module.asyncio, "to_thread", fake_to_thread),
            ):
                events = [
                    event
                    async for event in turn_module.run_turn(
                        "w_test",
                        "conversation-1",
                        "Analyze the sweep.",
                        sandbox="workspace-write",
                        role_runtimes={
                            "orchestrator": {
                                "model": "gpt-5.6-terra",
                                "effort": "high",
                            }
                        },
                    )
                ]

        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[1][1], "session-1")
        self.assertIn("previous call ended with a non-terminal `milestone`", prompts[1][0])
        self.assertIn(
            {
                "kind": "intermediate_output",
                "role": "orchestrator",
                "model": "gpt-5.6-terra",
                "effort": "high",
                "level": "milestone",
                "text": "Sweep ready.",
            },
            events,
        )
        self.assertEqual(
            events[-1],
            {"kind": "final", "outcome": "final_answer", "text": "TP4 wins."},
        )

    async def test_unparsed_answer_is_repaired_in_same_session(self) -> None:
        prompts: list[tuple[str, str | None]] = []

        async def fake_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        async def fake_run_codex(container, prompt, **kwargs):
            del container
            prompts.append((prompt, kwargs["session_id"]))
            if len(prompts) == 1:
                yield {
                    "kind": "session",
                    "role": "orchestrator",
                    "model": "gpt-5.6-terra",
                    "effort": "high",
                    "session_id": "session-1",
                }
                yield {
                    "kind": "final",
                    "text": "Analysis complete. Throughput rises with TP.",
                }
                return
            yield {
                "kind": "final",
                "text": (
                    '{"action":"final_answer",'
                    '"message":"Analysis complete. Throughput rises with TP.",'
                    '"task":""}'
                ),
            }

        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            with (
                patch.object(turn_module, "workspace_main_for", return_value=workspace),
                patch.object(turn_module, "prepare_workspace", return_value=workspace),
                patch.object(turn_module, "container_running", return_value=True),
                patch.object(turn_module, "ensure_container", return_value="container"),
                patch.object(turn_module, "write_managed_context"),
                patch.object(turn_module, "run_codex", fake_run_codex),
                patch.object(turn_module.asyncio, "to_thread", fake_to_thread),
            ):
                events = [
                    event
                    async for event in turn_module.run_turn(
                        "w_test",
                        "conversation-1",
                        "Analyze the sweep.",
                        sandbox="workspace-write",
                        role_runtimes={
                            "orchestrator": {
                                "model": "gpt-5.6-terra",
                                "effort": "high",
                            }
                        },
                    )
                ]

        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[1][1], "session-1")
        self.assertIn("Do not redo completed analysis", prompts[1][0])
        self.assertIn("Analysis complete. Throughput rises with TP.", prompts[1][0])
        self.assertEqual(
            events[-1],
            {
                "kind": "final",
                "outcome": "final_answer",
                "text": "Analysis complete. Throughput rises with TP.",
            },
        )


IMPLEMENTER_REPORT = (
    "Implemented and committed the batched event-loop delivery trial. "
    "Commit fbed215. 40 tests pass."
)


async def _run(fake_run_codex, *, agent_mode: str = "orchestrated") -> list[dict]:
    """Drain one turn with the docker/workspace/subprocess layers stubbed out."""

    async def fake_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    with TemporaryDirectory() as temporary_directory:
        workspace = Path(temporary_directory)
        with (
            patch.object(turn_module, "workspace_main_for", return_value=workspace),
            patch.object(turn_module, "prepare_workspace", return_value=workspace),
            patch.object(turn_module, "container_running", return_value=True),
            patch.object(turn_module, "ensure_container", return_value="container"),
            patch.object(turn_module, "write_managed_context"),
            patch.object(turn_module, "run_codex", fake_run_codex),
            patch.object(turn_module.asyncio, "to_thread", fake_to_thread),
        ):
            return [
                event
                async for event in turn_module.run_turn(
                    "w_test",
                    "conversation-1",
                    "Analyze the sweep.",
                    sandbox="workspace-write",
                    agent_mode=agent_mode,
                    role_runtimes={
                        driving_role_for_agent_mode(agent_mode): {
                            "model": "gpt-5.6-terra",
                            "effort": "high",
                        }
                    },
                )
            ]


class TransportFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_upstream_failure_skips_the_repair_loop(self) -> None:
        """A call that never reached the model has no decision to repair, and
        each repair round would spend another full reconnect cycle."""
        calls: list[str] = []

        async def fake_run_codex(container, prompt, **kwargs):
            del container, kwargs
            calls.append(prompt)
            yield {
                "kind": "final",
                "text": "(orchestrator produced no output: upstream returned 503)",
                "failure": {"code": "upstream_unavailable", "status": 503},
            }

        events = await _run(fake_run_codex)

        self.assertEqual(len(calls), 1)
        self.assertEqual(events[-1]["kind"], "final")
        self.assertEqual(
            events[-1]["failure"], {"code": "upstream_unavailable", "status": 503}
        )
        # A failed turn made no decision, so it reports no outcome.
        self.assertNotIn("outcome", events[-1])
        self.assertIn("503", events[-1]["text"])
        self.assertIn("not a model or parsing problem", events[-1]["text"])
        self.assertNotIn(
            "could not parse the orchestrator decision", events[-1]["text"]
        )

    async def test_rate_limit_keeps_its_own_code(self) -> None:
        async def fake_run_codex(container, prompt, **kwargs):
            del container, prompt, kwargs
            yield {
                "kind": "final",
                "text": "(orchestrator produced no output: upstream returned 429)",
                "failure": {"code": "upstream_rate_limited", "status": 429},
            }

        events = await _run(fake_run_codex)

        self.assertEqual(
            events[-1]["failure"], {"code": "upstream_rate_limited", "status": 429}
        )
        self.assertIn("rate limiting", events[-1]["text"])

    async def test_failure_after_a_round_reports_the_count_not_the_report(self) -> None:
        """The implementer's conclusion already renders as its own card. Quoting
        it into the assistant's own answer is what read as the two roles being
        confused."""
        calls: list[str] = []

        async def fake_run_codex(container, prompt, **kwargs):
            del container, kwargs
            calls.append(prompt)
            if len(calls) == 1:
                yield {
                    "kind": "final",
                    "text": '{"action":"delegate","message":"","task":"Batch it."}',
                }
                return
            if len(calls) == 2:
                yield {"kind": "final", "text": IMPLEMENTER_REPORT}
                return
            yield {
                "kind": "final",
                "text": "(orchestrator produced no output: upstream returned 503)",
                "failure": {"code": "upstream_unavailable", "status": 503},
            }

        events = await _run(fake_run_codex)

        self.assertIn({"kind": "implementer", "text": IMPLEMENTER_REPORT}, events)
        self.assertNotIn(IMPLEMENTER_REPORT, events[-1]["text"])
        self.assertNotIn("### Implementer Summary", events[-1]["text"])
        self.assertIn("1 implementer round completed", events[-1]["text"])

    async def test_implementer_failure_does_not_call_the_orchestrator_again(
        self,
    ) -> None:
        """Handing a placeholder summary back would spend one more call against
        the same unavailable gateway to conclude nothing."""
        calls: list[str] = []

        async def fake_run_codex(container, prompt, **kwargs):
            del container, kwargs
            calls.append(prompt)
            if len(calls) == 1:
                yield {
                    "kind": "final",
                    "text": '{"action":"delegate","message":"","task":"Batch it."}',
                }
                return
            yield {
                "kind": "final",
                "text": "(implementer produced no output: upstream returned 503)",
                "failure": {"code": "upstream_unavailable", "status": 503},
            }

        events = await _run(fake_run_codex)

        self.assertEqual(len(calls), 2)
        self.assertNotIn("implementer", [event["kind"] for event in events])
        self.assertEqual(
            events[-1]["failure"], {"code": "upstream_unavailable", "status": 503}
        )
        self.assertIn("The implementer could not run", events[-1]["text"])

    async def test_unparsed_decision_does_not_quote_implementer_summaries(self) -> None:
        """Bug B on the pre-existing path: exhausting the repair budget used to
        emit the implementer's report as the assistant's own answer."""
        calls: list[str] = []

        async def fake_run_codex(container, prompt, **kwargs):
            del container, kwargs
            calls.append(prompt)
            if len(calls) == 1:
                yield {
                    "kind": "final",
                    "text": '{"action":"delegate","message":"","task":"Batch it."}',
                }
                return
            if len(calls) == 2:
                yield {"kind": "final", "text": IMPLEMENTER_REPORT}
                return
            yield {"kind": "final", "text": "not a decision envelope at all"}

        events = await _run(fake_run_codex)

        self.assertNotIn(IMPLEMENTER_REPORT, events[-1]["text"])
        self.assertNotIn("### Implementer Summary", events[-1]["text"])
        self.assertIn("could not parse the orchestrator decision", events[-1]["text"])
        self.assertIn("1 implementer round completed", events[-1]["text"])


class SingleAgentTests(unittest.IsolatedAsyncioTestCase):
    """`agent_mode="single"` runs the same loop with one role and no delegation."""

    async def test_final_answer_costs_exactly_one_codex_call(self) -> None:
        calls: list[str] = []

        async def fake_run_codex(container, prompt, **kwargs):
            del container, kwargs
            calls.append(prompt)
            yield {
                "kind": "final",
                "text": '{"action":"final_answer","message":"Done."}',
            }

        events = await _run(fake_run_codex, agent_mode="single")

        self.assertEqual(len(calls), 1)
        self.assertIn("You are the VibeSim assistant.", calls[0])
        self.assertEqual(
            events[-1],
            {"kind": "final", "outcome": "final_answer", "text": "Done."},
        )

    async def test_the_driving_role_is_assistant_and_nothing_is_delegated(
        self,
    ) -> None:
        labels: list[str] = []

        async def fake_run_codex(container, prompt, **kwargs):
            del container, prompt
            labels.append(kwargs["label"])
            yield {"kind": "tool_call", "role": kwargs["label"], "text": "rg pattern"}
            yield {
                "kind": "final",
                "text": '{"action":"milestone","message":"Half way."}',
            }

        events = await _run(fake_run_codex, agent_mode="single")

        self.assertEqual(set(labels), {"assistant"})
        roles = {event["role"] for event in events if "role" in event}
        self.assertEqual(roles, {"assistant"})
        kinds = {event.get("kind") for event in events}
        self.assertNotIn("decision", kinds)
        self.assertNotIn("implementer", kinds)

    async def test_a_delegate_envelope_is_repaired_not_obeyed(self) -> None:
        """There is no implementer, so the decision must go back for repair and
        the repair budget must still bound the turn."""
        calls: list[str] = []

        async def fake_run_codex(container, prompt, **kwargs):
            del container, kwargs
            calls.append(prompt)
            yield {
                "kind": "final",
                "text": '{"action":"delegate","message":"","task":"Change it."}',
            }

        events = await _run(fake_run_codex, agent_mode="single")

        self.assertEqual(
            len(calls), turn_module.MAX_ORCHESTRATOR_DECISION_REPAIRS + 1
        )
        for prompt in calls[1:]:
            self.assertIn("could not be parsed", prompt)
            self.assertIn("There is no `delegate` action", prompt)
        self.assertEqual(events[-1]["kind"], "final")
        self.assertNotIn("implementer round", events[-1]["text"])


if __name__ == "__main__":
    unittest.main()
