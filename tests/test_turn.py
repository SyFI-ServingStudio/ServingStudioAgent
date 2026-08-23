from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.codex_runtime import turn as turn_module
from backend.codex_runtime.config import (
    ASSISTANT_SCHEMA_IN_CONTAINER,
    CODEXDS_MODEL,
    IMPLEMENTER_SCHEMA_IN_CONTAINER,
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
                    "text": ('{"action":"final_answer","message":"Done.","task":""}'),
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
                        '{"action":"milestone","message":"Sweep ready.","task":""}'
                    ),
                }
                return
            yield {
                "kind": "final",
                "text": ('{"action":"final_answer","message":"TP4 wins.","task":""}'),
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
        self.assertIn(
            "previous call ended with a non-terminal `milestone`", prompts[1][0]
        )
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

    async def test_a_checkpoint_message_precedes_its_own_usage_event(self) -> None:
        """The UI closes a role card on `usage`, so the order decides the round.

        A checkpoint decision is parsed only after the call has ended, so its
        message is emitted last. Forwarding `usage` on arrival would close the
        card first and push that message — the call's own conclusion — into the
        next round's card, which reads as one call cut across two rounds.
        """
        calls = 0

        async def fake_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        async def fake_run_codex(container, prompt, **kwargs):
            nonlocal calls
            del container, prompt
            calls += 1
            yield {
                "kind": "usage",
                "role": "orchestrator",
                "model": "gpt-5.6-terra",
                "effort": "high",
                "duration_ms": 10,
                "tokens": {},
            }
            yield {
                "kind": "final",
                "text": (
                    '{"action":"progress","message":"Sweep launched.","task":""}'
                    if calls == 1
                    else '{"action":"final_answer","message":"TP4 wins.","task":""}'
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
                            "orchestrator": {"model": "gpt-5.6-terra", "effort": "high"}
                        },
                    )
                ]

        # `tool_call` is narration for the wait, not part of the transcript.
        transcript = [event for event in events if event["kind"] != "tool_call"]
        self.assertEqual(
            [event["kind"] for event in transcript],
            ["intermediate_output", "usage", "usage", "final"],
        )
        self.assertEqual(transcript[0]["text"], "Sweep launched.")

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


async def _run(
    fake_run_codex,
    *,
    agent_mode: str = "orchestrated",
    sandbox: str = "workspace-write",
    sessions: dict[str, str] | None = None,
    resume_role: str | None = None,
    role_runtimes: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
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
                    sandbox=sandbox,
                    sessions=sessions,
                    agent_mode=agent_mode,
                    resume_role=resume_role,
                    role_runtimes=role_runtimes
                    or {
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

        self.assertEqual(len(calls), turn_module.MAX_ORCHESTRATOR_DECISION_REPAIRS + 1)
        for prompt in calls[1:]:
            self.assertIn("could not be parsed", prompt)
            self.assertIn("There is no `delegate` action", prompt)
        self.assertEqual(events[-1]["kind"], "final")
        self.assertNotIn("implementer round", events[-1]["text"])


class ResumeInterruptedImplementerTests(unittest.IsolatedAsyncioTestCase):
    """`resume_role="implementer"` reopens the turn at the role the user stopped.

    The driving role needs no equivalent: it resumes its own session on every
    turn already, so only the implementer can be stranded by an interrupt.
    """

    @staticmethod
    def _implementer_then_orchestrator(calls: list[tuple[str, str]]):
        async def fake_run_codex(container, prompt, **kwargs):
            del container
            label = kwargs["label"]
            calls.append((label, prompt))
            if label == "implementer":
                yield {"kind": "final", "text": "Reran it with uv."}
            else:
                yield {
                    "kind": "final",
                    "text": '{"action":"final_answer","message":"Done."}',
                }

        return fake_run_codex

    async def test_the_implementer_runs_first_then_hands_back_to_the_driver(
        self,
    ) -> None:
        calls: list[tuple[str, str]] = []

        events = await _run(
            self._implementer_then_orchestrator(calls),
            sessions={"implementer": "session-impl"},
            resume_role="implementer",
        )

        self.assertEqual([label for label, _ in calls], ["implementer", "orchestrator"])
        # The user's words arrive as a correction to the task already in the
        # session, not as a fresh delegation.
        self.assertIn("interrupted you while you were working", calls[0][1])
        self.assertIn("Analyze the sweep.", calls[0][1])
        self.assertNotIn("Task:\nAnalyze the sweep.", calls[0][1])
        # ...and its summary comes back through the normal handoff, so the turn
        # still ends with the orchestrator's answer.
        self.assertIn("Reran it with uv.", calls[1][1])
        self.assertEqual(
            events[-1],
            {"kind": "final", "outcome": "final_answer", "text": "Done."},
        )

    async def test_the_implementer_session_is_resumed_not_started(self) -> None:
        resumed: list[str | None] = []

        async def fake_run_codex(container, prompt, **kwargs):
            del container, prompt
            if kwargs["label"] == "implementer":
                resumed.append(kwargs["session_id"])
                yield {"kind": "final", "text": "Reran it with uv."}
            else:
                yield {
                    "kind": "final",
                    "text": '{"action":"final_answer","message":"Done."}',
                }

        await _run(
            fake_run_codex,
            sessions={"implementer": "session-impl"},
            resume_role="implementer",
        )

        self.assertEqual(resumed, ["session-impl"])

    async def test_a_reply_to_the_user_ends_the_turn_without_the_driver(
        self,
    ) -> None:
        """Asking the implementer a question is not delegated work: routing its
        answer through the orchestrator would spend a call to have a third party
        repeat it."""
        calls: list[str] = []

        async def fake_run_codex(container, prompt, **kwargs):
            del container, prompt
            calls.append(kwargs["label"])
            yield {
                "kind": "final",
                "text": '{"action":"reply_user","message":"Reprofiling the GEMM."}',
            }

        events = await _run(
            fake_run_codex,
            sessions={"implementer": "session-impl"},
            resume_role="implementer",
        )

        self.assertEqual(calls, ["implementer"])
        self.assertEqual(
            events[-1],
            {
                "kind": "final",
                "outcome": "final_answer",
                "text": "Reprofiling the GEMM.",
                # Names the answerer, so the next message continues with it.
                "role": "implementer",
            },
        )
        # A reply is not a summary, so it neither becomes a handoff card nor
        # joins the summaries the final answer is composed against.
        self.assertNotIn("implementer", [event["kind"] for event in events])

    async def test_a_delegated_task_cannot_reply_to_the_user(self) -> None:
        """Nothing in a delegated prompt came from the user, so `reply_user`
        there would end the turn on an answer to a question nobody asked. It is
        demoted to a summary rather than obeyed."""
        calls: list[str] = []

        async def fake_run_codex(container, prompt, **kwargs):
            del container, prompt
            label = kwargs["label"]
            calls.append(label)
            if label == "implementer":
                yield {
                    "kind": "final",
                    "text": '{"action":"reply_user","message":"Reran it with uv."}',
                }
            elif len(calls) == 1:
                yield {
                    "kind": "final",
                    "text": '{"action":"delegate","task":"Rerun the sweep."}',
                }
            else:
                yield {
                    "kind": "final",
                    "text": '{"action":"final_answer","message":"Done."}',
                }

        events = await _run(fake_run_codex)

        self.assertEqual(calls, ["orchestrator", "implementer", "orchestrator"])
        self.assertIn(
            {"kind": "implementer", "text": "Reran it with uv."},
            events,
        )
        self.assertEqual(
            events[-1],
            {"kind": "final", "outcome": "final_answer", "text": "Done."},
        )

    async def test_the_implementer_envelope_is_constrained_except_on_deepseek(
        self,
    ) -> None:
        """Same carve-out the driver already has: vLLM's constrained text branch
        can end a DeepSeek turn early, so that family keeps the prompt contract
        alone."""

        async def schema_for(model_id: str) -> str | None:
            captured: str | None = None

            async def fake_run_codex(container, prompt, **kwargs):
                nonlocal captured
                del container, prompt
                if kwargs["label"] == "implementer":
                    captured = kwargs["output_schema"]
                    yield {
                        "kind": "final",
                        "text": '{"action":"reply_user","message":"Still running."}',
                    }
                    return
                yield {
                    "kind": "final",
                    "text": '{"action":"final_answer","message":"Done."}',
                }

            await _run(
                fake_run_codex,
                sessions={"implementer": "session-impl"},
                resume_role="implementer",
                role_runtimes={
                    "orchestrator": {"model": "gpt-5.6-terra", "effort": "high"},
                    "implementer": {
                        "model": model_id,
                        "effort": "max" if model_id == CODEXDS_MODEL else "high",
                    },
                },
            )
            return captured

        self.assertEqual(
            await schema_for("gpt-5.6-terra"), IMPLEMENTER_SCHEMA_IN_CONTAINER
        )
        self.assertIsNone(await schema_for(CODEXDS_MODEL))

    async def test_a_free_text_summary_still_reaches_the_orchestrator(self) -> None:
        """The envelope is new; transcripts and models that predate it are not.
        Anything unparseable stays a summary, which is what it always was."""
        calls: list[tuple[str, str]] = []

        events = await _run(
            self._implementer_then_orchestrator(calls),
            sessions={"implementer": "session-impl"},
            resume_role="implementer",
        )

        self.assertEqual([label for label, _ in calls], ["implementer", "orchestrator"])
        self.assertIn("Reran it with uv.", calls[1][1])
        self.assertEqual(events[-1]["text"], "Done.")

    async def test_a_stale_request_falls_back_to_an_ordinary_turn(self) -> None:
        """Each condition is a way the request can be stale, never an error: the
        driving role still has the conversation and can re-delegate."""
        stale_cases = {
            "no implementer session": {"sessions": {}, "resume_role": "implementer"},
            "single agent mode": {
                "sessions": {"implementer": "session-impl"},
                "resume_role": "implementer",
                "agent_mode": "single",
            },
            "read-only turn": {
                "sessions": {"implementer": "session-impl"},
                "resume_role": "implementer",
                "sandbox": "read-only",
            },
            "driver was interrupted": {
                "sessions": {"implementer": "session-impl"},
                "resume_role": "orchestrator",
            },
        }
        for name, arguments in stale_cases.items():
            with self.subTest(name):
                calls: list[tuple[str, str]] = []
                await _run(self._implementer_then_orchestrator(calls), **arguments)
                self.assertEqual(
                    calls[0][0],
                    driving_role_for_agent_mode(
                        str(arguments.get("agent_mode", "orchestrated"))
                    ),
                )


if __name__ == "__main__":
    unittest.main()
