import json
import logging
import unittest
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.runtime_fixtures import docker_execution
from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import AgentMode, Role
from vibesim_agent.domain.turns import Outcome, TurnInput, TurnResult
from vibesim_agent.prompts.render import Prompts
from vibesim_agent.providers.base import Model, Provider
from vibesim_agent.providers.registry import ProviderRegistry
from vibesim_agent.services.driver import ConversationDriver
from vibesim_agent.services.turn import TurnService, TurnStorage
from vibesim_agent.settings import ProviderSettings
from vibesim_agent.storage.conversations import Conversations
from vibesim_agent.storage.database import Database
from vibesim_agent.storage.sessions import Sessions
from vibesim_agent.storage.turns import Turns


class ScriptedAdapter:
    adapter_id = "scripted"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.closed = 0

    async def run(self, request):
        self.requests.append(request)
        try:
            yield {"kind": "role_start", "role": "assistant"}
            yield {"kind": "session", "role": "assistant", "session_id": "saved"}
            yield {"kind": "role_ready", "role": "assistant"}
            yield {"kind": "usage", "role": "assistant"}
            reply = next(self.replies)
            yield reply if isinstance(reply, dict) else {"kind": "final", "text": reply}
        finally:
            self.closed += 1


class SingleDriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_schema_path_comes_from_the_execution(self):
        # Not from the driver: the same contract file is a read-only mount
        # target in a container and a state-root path on the host, and Codex is
        # handed whichever one this turn produced as `--output-schema`.
        driver, adapter = self.driver(
            [json.dumps({"action": "final_answer", "message": "done"})]
        )
        directory = Path(self.enterContext(TemporaryDirectory()))
        database = Database.create(directory / "workspace.sqlite")
        store = TurnStorage(
            Conversations(database), Sessions(database), Turns(database)
        )
        store.conversations.create(
            "c", agent_mode=AgentMode.SINGLE, runtimes=self.request().runtimes
        )
        service = TurnService(
            lambda _: store, driver, logger=logging.getLogger("schema-test")
        )
        self.addAsyncCleanup(service.close)
        await service.wait(service.start("w", "c", "question"))
        [request] = adapter.requests
        self.assertEqual(
            request.output_schema, Path("/contracts/assistant.schema.json")
        )

    async def test_every_prompt_of_a_turn_names_the_paths_that_turn_can_see(self):
        driver, adapter = self.driver(
            [
                "not JSON",
                json.dumps({"action": "final_answer", "message": "done"}),
            ]
        )
        host = Prompts.prepare(
            Path(self.enterContext(TemporaryDirectory())), workspace=Path("/repo/wt")
        )
        driver.prompts_for = lambda request: host.bound(autonomous=request.autonomous)
        [event async for event in driver.run(self.request())]
        # The repair prompt too, not only the first: each one restates the
        # contract and the plan files by path.
        self.assertEqual(len(adapter.requests), 2)
        for request in adapter.requests:
            self.assertNotIn("/workspace", request.prompt)
            self.assertIn("`/repo/wt/c_plan.md`", request.prompt)

    async def test_service_driver_provider_storage_roundtrip_and_resume(self):
        driver, adapter = self.driver(
            [
                json.dumps({"action": "final_answer", "message": "first answer"}),
                json.dumps(
                    {"action": "request_user_input", "message": "next question"}
                ),
            ]
        )
        directory = Path(self.enterContext(TemporaryDirectory()))
        database = Database.create(directory / "workspace.sqlite")
        store = TurnStorage(
            Conversations(database), Sessions(database), Turns(database)
        )
        store.conversations.create(
            "c", agent_mode=AgentMode.SINGLE, runtimes=self.request().runtimes
        )
        service = TurnService(
            lambda _: store, driver, logger=logging.getLogger("pipeline-test")
        )
        self.addAsyncCleanup(service.close)
        for text, expected in (("first", Outcome.ANSWER), ("follow up", Outcome.INPUT)):
            handle = service.start("w", "c", text)
            self.assertEqual((await service.wait(handle)).outcome, expected)
            replay = [
                event
                async for event in service.stream("w", "c", handle.request.turn_id)
            ]
            self.assertEqual(replay[-1]["kind"], "done")
            self.assertEqual(replay[-1]["payload"]["outcome"], expected.value)
        self.assertEqual(
            [request.session_id for request in adapter.requests], [None, "saved"]
        )
        self.assertEqual(
            [message.content for message in store.conversations.messages("c")],
            ["first", "first answer", "follow up", "next question"],
        )

    def driver(self, replies):
        adapter = ScriptedAdapter(replies)
        registry = ProviderRegistry()
        model = Model("model", "Model", ("high",), "high")
        registry.register(
            Provider(
                "test",
                "Test",
                adapter,
                ProviderSettings(model="model", effort="high"),
                "scope",
                lambda: (model,),
            )
        )
        root = Path(self.enterContext(TemporaryDirectory()))

        async def prepare(request):
            # An execution rather than a name: the driver reads the schema
            # directory off it, because that path differs by mode.
            return replace(docker_execution(), schema_directory="/contracts")

        async def before_call(request):
            pass

        return ConversationDriver(
            registry,
            Prompts.prepare(root),
            prepare=prepare,
            before_call=before_call,
        ), adapter

    def request(self):
        return TurnInput(
            "w",
            "c",
            "t",
            "question",
            AgentMode.SINGLE,
            {Role.ASSISTANT: RoleRuntime("test", "scope", "model", "high", "default")},
            {},
            "",
        )

    async def test_repair_then_checkpoint_then_input_preserves_session_and_usage_order(
        self,
    ):
        driver, adapter = self.driver(
            [
                "not JSON",
                json.dumps({"action": "progress", "message": "working"}),
                json.dumps({"action": "request_user_input", "message": "which model?"}),
            ]
        )
        events = [event async for event in driver.run(self.request())]
        self.assertEqual(events[-1], TurnResult(Outcome.INPUT, "which model?"))
        self.assertEqual(
            [request.session_id for request in adapter.requests],
            [None, "saved", "saved"],
        )
        checkpoint = next(
            i
            for i, event in enumerate(events)
            if isinstance(event, dict) and event["kind"] == "intermediate_output"
        )
        self.assertEqual(events[checkpoint + 1]["kind"], "usage")
        self.assertEqual(adapter.closed, 3)

    async def test_transport_failure_is_not_retried(self):
        driver, adapter = self.driver(
            [
                {
                    "kind": "final",
                    "text": "upstream unavailable",
                    "failure": {"code": "upstream_unavailable", "status": 503},
                }
            ]
        )
        events = [event async for event in driver.run(self.request())]
        self.assertEqual(events[-1].outcome, Outcome.FAILED)
        self.assertEqual(len(adapter.requests), 1)

    async def test_continuation_budget_stops_after_four_calls(self):
        driver, adapter = self.driver(
            [json.dumps({"action": "progress", "message": "still working"})] * 4
        )
        events = [event async for event in driver.run(self.request())]
        self.assertEqual(events[-1].outcome, Outcome.FAILED)
        self.assertEqual(len(adapter.requests), 4)

    async def test_missing_final_is_runtime_failure_without_repair(self):
        driver, adapter = self.driver([{"kind": "usage", "role": "assistant"}])
        events = [event async for event in driver.run(self.request())]
        self.assertEqual(events[-1].outcome, Outcome.FAILED)
        self.assertEqual(
            events[-1].metadata["failure"]["code"], "agent_runtime_failure"
        )
        self.assertEqual(len(adapter.requests), 1)

    async def test_repair_budget_and_single_delegation_rejection(self):
        driver, adapter = self.driver(
            [json.dumps({"action": "delegate", "task": "work"})] * 3
        )
        events = [event async for event in driver.run(self.request())]
        self.assertEqual(events[-1].outcome, Outcome.FAILED)
        self.assertEqual(len(adapter.requests), 3)

    async def test_closing_driver_closes_nested_provider_immediately(self):
        driver, adapter = self.driver([])
        async with aclosing(driver.run(self.request())) as events:
            self.assertEqual((await anext(events))["kind"], "role_start")
            self.assertEqual(adapter.requests, [])
            self.assertEqual((await anext(events))["kind"], "session")
        self.assertEqual(adapter.closed, 1)
