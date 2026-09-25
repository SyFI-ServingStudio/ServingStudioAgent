import asyncio
import json
import logging
import unittest
from dataclasses import replace
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.runtime_fixtures import docker_execution
from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import AgentMode, Role, Sandbox
from vibesim_agent.domain.turns import Outcome, TurnInput
from vibesim_agent.prompts.render import Prompts
from vibesim_agent.providers.base import Model, OutputMode, Provider
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.services.driver import ConversationDriver
from vibesim_agent.services.turn import TurnService, TurnStorage
from vibesim_agent.settings import ProviderSettings
from vibesim_agent.storage.conversations import Conversations
from vibesim_agent.storage.database import Database
from vibesim_agent.storage.sessions import Sessions
from vibesim_agent.storage.turns import Turns


class ScriptedRoles:
    adapter_id = "test"

    def __init__(self, script):
        self.script = iter(script)
        self.requests = []

    async def run(self, request):
        expected, response = next(self.script)
        if request.role != expected:
            raise AssertionError(f"expected {expected}, got {request.role}")
        self.requests.append(request)
        yield {"kind": "role_start", "role": request.role.value}
        yield {
            "kind": "session",
            "role": request.role.value,
            "session_id": f"saved-{request.role.value}",
        }
        yield {"kind": "role_ready", "role": request.role.value}
        yield {"kind": "usage", "role": request.role.value}
        if "failure" in response:
            yield {"kind": "final", **response}
        else:
            yield {"kind": "final", "text": json.dumps(response)}


def delegate(task="inspect"):
    return {"action": "delegate", "task": task}


def answer(text="done"):
    return {"action": "final_answer", "message": text}


class OrchestratedDriverTests(unittest.IsolatedAsyncioTestCase):
    def setup_driver(self, script):
        root = Path(self.enterContext(TemporaryDirectory()))
        adapter = ScriptedRoles(script)
        registry = ProviderRegistry()
        runtimes = {}
        for role in (Role.ORCHESTRATOR, Role.IMPLEMENTER):
            provider_id = "primary" if role is Role.ORCHESTRATOR else "secondary"
            model = Model(
                provider_id,
                provider_id,
                ("high",),
                "high",
                output_mode=OutputMode.STRUCTURED
                if role is Role.ORCHESTRATOR
                else OutputMode.PROMPT,
            )
            registry.register(
                Provider(
                    provider_id,
                    provider_id,
                    adapter,
                    ProviderSettings(model=provider_id, effort="high"),
                    provider_id + ":scope",
                    lambda model=model: (model,),
                )
            )
            runtimes[role] = RoleRuntime(
                provider_id, provider_id + ":scope", provider_id, "high", "default"
            )

        async def prepare(request):
            # An execution rather than a name: the driver reads the schema
            # directory off it, because that path differs by mode.
            return replace(docker_execution(), schema_directory="/contracts")

        async def before_call(request):
            pass

        driver = ConversationDriver(
            registry,
            Prompts.prepare(root / "prompts"),
            prepare=prepare,
            before_call=before_call,
        )
        request = TurnInput(
            "w", "c", "t", "question", AgentMode.ORCHESTRATED, runtimes, {}, ""
        )
        return root, driver, adapter, request

    async def test_two_delegated_turns_keep_both_sessions_and_mixed_capabilities(self):
        root, driver, adapter, request = self.setup_driver(
            [
                (Role.ORCHESTRATOR, delegate()),
                (Role.IMPLEMENTER, answer("summary")),
                (Role.ORCHESTRATOR, answer()),
            ]
            * 2
        )
        database = Database.create(root / "workspace.sqlite")
        store = TurnStorage(
            Conversations(database), Sessions(database), Turns(database)
        )
        store.conversations.create("c", runtimes=request.runtimes)
        service = TurnService(
            lambda _: store, driver, logger=logging.getLogger("dual-test")
        )
        self.addAsyncCleanup(service.close)
        for _ in range(2):
            handle = service.start("w", "c", "question")
            result = await service.wait(handle)
            self.assertEqual(result.outcome, Outcome.ANSWER)
            self.assertEqual(result.metadata["implementer_summaries"], ["summary"])
        self.assertEqual(
            [call.session_id for call in adapter.requests],
            [
                None,
                None,
                "saved-orchestrator",
                "saved-orchestrator",
                "saved-implementer",
                "saved-orchestrator",
            ],
        )
        self.assertEqual(
            [call.structured_output for call in adapter.requests],
            [True, False, True] * 2,
        )
        self.assertIn("Implementer summary:\nsummary", adapter.requests[2].prompt)
        self.assertEqual(
            [message.role for message in store.conversations.messages("c")],
            ["user", "assistant"] * 2,
        )

    async def test_user_steer_direct_reply_keeps_implementer_landing(self):
        _, driver, adapter, request = self.setup_driver(
            [(Role.IMPLEMENTER, {"action": "reply_user", "message": "clarification"})]
        )
        request = replace(
            request,
            resume_role="implementer",
            sessions={Role.IMPLEMENTER: "old-session"},
        )
        events = [event async for event in driver.run(request)]
        self.assertEqual(events[-1].resume_role, Role.IMPLEMENTER)
        self.assertIn("user: question", adapter.requests[0].prompt)
        self.assertEqual(len(adapter.requests), 1)

    async def test_delegated_reply_user_is_a_summary_not_a_direct_reply(self):
        _, driver, adapter, request = self.setup_driver(
            [
                (Role.ORCHESTRATOR, delegate()),
                (Role.IMPLEMENTER, {"action": "reply_user", "message": "summary"}),
                (Role.ORCHESTRATOR, answer("reviewed")),
            ]
        )
        events = [event async for event in driver.run(request)]
        self.assertEqual(events[-1].text, "reviewed")
        self.assertIsNone(events[-1].resume_role)
        self.assertEqual(len(adapter.requests), 3)

    async def test_steer_without_implementer_session_resumes_orchestrator(self):
        _, driver, adapter, request = self.setup_driver(
            [
                (Role.ORCHESTRATOR, answer("reviewed")),
            ]
        )
        request = replace(
            request,
            resume_role="implementer",
            sessions={Role.ORCHESTRATOR: "old-orchestrator"},
        )
        events = [event async for event in driver.run(request)]
        self.assertEqual(events[-1].outcome, Outcome.ANSWER)
        self.assertEqual(events[-1].text, "reviewed")
        self.assertIsNone(events[-1].resume_role)
        self.assertEqual(len(adapter.requests), 1)
        self.assertEqual(adapter.requests[0].role, Role.ORCHESTRATOR)
        self.assertEqual(adapter.requests[0].session_id, "old-orchestrator")

    async def test_read_only_prevents_implementer_even_with_resume_marker(self):
        _, driver, adapter, request = self.setup_driver(
            [(Role.ORCHESTRATOR, delegate())]
        )
        request = replace(
            request,
            sandbox=Sandbox.READ_ONLY,
            resume_role="implementer",
            sessions={Role.IMPLEMENTER: "saved"},
        )
        events = [event async for event in driver.run(request)]
        self.assertEqual(len(adapter.requests), 1)
        self.assertIn("read-only", events[-1].text)

    async def test_failure_preserves_completed_summaries_without_gateway_retry(self):
        _, driver, adapter, request = self.setup_driver(
            [
                (Role.ORCHESTRATOR, delegate()),
                (Role.IMPLEMENTER, answer("first done")),
                (Role.ORCHESTRATOR, delegate("second")),
                (
                    Role.IMPLEMENTER,
                    {
                        "text": "gateway down",
                        "failure": {"code": "upstream_unavailable", "status": 503},
                    },
                ),
            ]
        )
        events = [event async for event in driver.run(request)]
        self.assertEqual(events[-1].outcome, Outcome.FAILED)
        self.assertEqual(events[-1].metadata["implementer_summaries"], ["first done"])
        self.assertEqual(len(adapter.requests), 4)

    async def test_each_implementer_round_restores_the_checkpoint_budget(self):
        milestone = {"action": "milestone", "message": "round landed"}
        rounds = []
        for index in range(4):
            rounds += [
                (Role.ORCHESTRATOR, delegate(f"round {index}")),
                (Role.IMPLEMENTER, answer(f"round {index} done")),
                (Role.ORCHESTRATOR, milestone),
                (Role.ORCHESTRATOR, milestone),
            ]
        _, driver, adapter, request = self.setup_driver(
            [*rounds, (Role.ORCHESTRATOR, answer("all done"))]
        )
        events = [event async for event in driver.run(request)]
        self.assertEqual(events[-1].outcome, Outcome.ANSWER, events[-1].text)
        self.assertEqual(len(adapter.requests), 17)

    async def test_consecutive_checkpoints_still_fail_after_a_round(self):
        milestone = {"action": "milestone", "message": "still here"}
        _, driver, adapter, request = self.setup_driver(
            [
                (Role.ORCHESTRATOR, delegate()),
                (Role.IMPLEMENTER, answer("first done")),
                *[(Role.ORCHESTRATOR, milestone)] * 4,
            ]
        )
        events = [event async for event in driver.run(request)]
        self.assertEqual(events[-1].outcome, Outcome.FAILED)
        self.assertEqual(
            events[-1].metadata["failure"], {"code": "agent_checkpoint_loop"}
        )
        self.assertEqual(len(adapter.requests), 6)

    async def test_implementer_checkpoint_resumes_the_implementer_not_a_handoff(self):
        _, driver, adapter, request = self.setup_driver(
            [
                (Role.ORCHESTRATOR, delegate("build it")),
                (Role.IMPLEMENTER, {"action": "milestone", "message": "half built"}),
                (Role.IMPLEMENTER, {"action": "progress", "message": "testing"}),
                (Role.IMPLEMENTER, answer("built")),
                (Role.ORCHESTRATOR, answer("reviewed")),
            ]
        )
        events = [event async for event in driver.run(request)]
        self.assertEqual(events[-1].outcome, Outcome.ANSWER)
        self.assertEqual(events[-1].metadata["implementer_summaries"], ["built"])
        notes = [
            (event["level"], event["text"])
            for event in events[:-1]
            if event["kind"] == "intermediate_output"
        ]
        self.assertEqual(notes, [("milestone", "half built"), ("progress", "testing")])
        handoffs = [event["text"] for event in events[:-1] if event["kind"] == "implementer"]
        self.assertEqual(handoffs, ["built"])
        resumed = adapter.requests[2]
        self.assertEqual(resumed.role, Role.IMPLEMENTER)
        self.assertEqual(resumed.session_id, "saved-implementer")
        self.assertIn("non-terminal `milestone` update", resumed.prompt)
        self.assertIn("half built", resumed.prompt)
        self.assertNotIn("reply_user` if", resumed.prompt)
        self.assertNotIn("\nuser: ", resumed.prompt)
        self.assertIn("Implementer summary:\nbuilt", adapter.requests[4].prompt)

    async def test_steered_implementer_checkpoint_keeps_the_user_line(self):
        _, driver, adapter, request = self.setup_driver(
            [
                (Role.IMPLEMENTER, {"action": "progress", "message": "checking"}),
                (Role.IMPLEMENTER, {"action": "reply_user", "message": "it is X"}),
            ]
        )
        request = replace(
            request,
            resume_role="implementer",
            sessions={Role.IMPLEMENTER: "old-session"},
        )
        events = [event async for event in driver.run(request)]
        self.assertEqual(events[-1].text, "it is X")
        self.assertEqual(events[-1].resume_role, Role.IMPLEMENTER)
        self.assertIn("user: question", adapter.requests[1].prompt)
        self.assertIn("reply_user` if", adapter.requests[1].prompt)

    async def test_implementer_checkpoints_fail_after_the_budget(self):
        stop = {"action": "progress", "message": "still going"}
        _, driver, adapter, request = self.setup_driver(
            [
                (Role.ORCHESTRATOR, {"action": "milestone", "message": "planned"}),
                (Role.ORCHESTRATOR, delegate()),
                *[(Role.IMPLEMENTER, stop)] * 4,
            ]
        )
        events = [event async for event in driver.run(request)]
        self.assertEqual(events[-1].outcome, Outcome.FAILED)
        self.assertEqual(
            events[-1].metadata["failure"], {"code": "agent_checkpoint_loop"}
        )
        self.assertIn("implementer", events[-1].text)
        # The orchestrator's checkpoint before delegating is not charged to
        # the implementer round: all four implementer stops were allowed to run.
        self.assertEqual(len(adapter.requests), 6)

    async def test_role_start_precedes_awaiting_initial_preparation(self):
        root, driver, adapter, request = self.setup_driver([])
        entered = asyncio.Event()

        async def before_call(call):
            entered.set()
            await asyncio.Event().wait()

        driver.before_call = before_call
        database = Database.create(root / "workspace.sqlite")
        store = TurnStorage(
            Conversations(database), Sessions(database), Turns(database)
        )
        store.conversations.create("c", runtimes=request.runtimes)
        service = TurnService(
            lambda _: store,
            driver,
            safe_interrupt_timeout=0.01,
            logger=logging.getLogger("handoff-test"),
        )
        self.addAsyncCleanup(service.close)
        handle = service.start("w", "c", "question")
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(handle.role, "orchestrator")
        self.assertFalse(handle.ready)
        service.cancel("w", "c", handle.request.turn_id)
        result = await asyncio.wait_for(service.wait(handle), 1)
        self.assertEqual(result.outcome, Outcome.CANCELLED)
        self.assertIsNone(result.resume_role)
        self.assertEqual(adapter.requests, [])

    async def test_handoff_preparation_clears_previous_role_readiness(self):
        root, driver, adapter, request = self.setup_driver(
            [
                (Role.ORCHESTRATOR, delegate()),
            ]
        )
        entered = asyncio.Event()

        async def before_call(call):
            if call.role is Role.IMPLEMENTER:
                entered.set()
                await asyncio.Event().wait()

        driver.before_call = before_call
        database = Database.create(root / "workspace.sqlite")
        store = TurnStorage(
            Conversations(database), Sessions(database), Turns(database)
        )
        store.conversations.create("c", runtimes=request.runtimes)
        service = TurnService(
            lambda _: store,
            driver,
            safe_interrupt_timeout=0.01,
            logger=logging.getLogger("handoff-test"),
        )
        self.addAsyncCleanup(service.close)
        handle = service.start("w", "c", "question")
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(handle.role, "implementer")
        self.assertFalse(handle.ready)
        records = store.turns.events("c", handle.request.turn_id)
        self.assertTrue(
            any(
                event["kind"] == "role_ready"
                and event["payload"]["role"] == "orchestrator"
                for event in records
            )
        )
        service.cancel("w", "c", handle.request.turn_id)
        result = await asyncio.wait_for(service.wait(handle), 1)
        self.assertEqual(result.outcome, Outcome.CANCELLED)
        self.assertIsNone(result.resume_role)
        self.assertEqual([call.role for call in adapter.requests], [Role.ORCHESTRATOR])
        self.assertEqual(store.conversations.get("c")["interrupted_role"], "")
