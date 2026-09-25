import asyncio
import json
import logging
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx

from tests.runtime_fixtures import docker_execution
from tests.http_support import LiveRequest, sse_events
from tests.test_analyzer_context import dictionary
from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import AgentMode, Role
from vibesim_agent.main import create_app
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


class Adapter:
    adapter_id = "test"

    def __init__(self):
        self.requests = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def run(self, request):
        self.requests.append(request)
        yield {"kind": "role_start", "role": "assistant"}
        yield {"kind": "session", "role": "assistant", "session_id": "saved"}
        yield {"kind": "role_ready", "role": "assistant"}
        self.entered.set()
        await self.release.wait()
        yield {
            "kind": "final",
            "text": json.dumps(
                {
                    "action": "final_answer",
                    "message": "Use `exp.tp2.rate20.throughput`.",
                }
            ),
        }


class HttpV2Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        database = Database.create(root / "workspace.sqlite")
        self.store = TurnStorage(
            Conversations(database), Sessions(database), Turns(database)
        )
        self.runtime = RoleRuntime("test", "scope", "model", "high", "default")
        for cid in ("c", "other"):
            self.store.conversations.create(
                cid,
                agent_mode=AgentMode.SINGLE,
                runtimes={Role.ASSISTANT: self.runtime},
            )
        self.adapter = Adapter()
        providers = ProviderRegistry()
        providers.register(
            Provider(
                "test",
                "Test",
                self.adapter,
                ProviderSettings(model="model", effort="high"),
                "scope",
                lambda: (Model("model", "Model", ("high",), "high"),),
            )
        )

        async def prepare(request):
            # An execution rather than a name: the driver reads the schema
            # directory off it, because that path differs by mode.
            return replace(docker_execution(), schema_directory="/contracts")

        async def before_call(request):
            pass

        prompts = Prompts.prepare(root / "prompts")
        driver = ConversationDriver(
            providers,
            prompts,
            prepare=prepare,
            before_call=before_call,
        )

        def storage(workspace_id):
            if workspace_id != "w":
                raise KeyError(workspace_id)
            return self.store

        self.turns = TurnService(
            storage,
            driver,
            logger=logging.getLogger("http-v2-test"),
            fingerprint=lambda request: prompts.fingerprint(
                mode=request.mode,
                autonomous=request.autonomous,
                runtimes=request.runtimes,
            ),
        )
        self.app = create_app(self.turns)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.addAsyncCleanup(self.turns.close)
        self.path = "/api/agent/v1/workspaces/w/conversations/c"

    async def test_quiet_turn_stream_carries_keepalive_comments(self):
        self.adapter.release.clear()
        with patch("vibesim_agent.api.conversations.KEEPALIVE_SECONDS", 0.05):
            response_task = asyncio.create_task(
                self.client.post(self.path + "/messages", json={"text": "slow work"})
            )
            try:
                await asyncio.wait_for(self.adapter.entered.wait(), 2)
                await asyncio.sleep(0.3)
                self.adapter.release.set()
                response = await asyncio.wait_for(response_task, 3)
            finally:
                self.adapter.release.set()
                if not response_task.done():
                    response_task.cancel()
                await asyncio.gather(response_task, return_exceptions=True)
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.text.count(": keepalive\n\n"), 2)
        kinds = [kind for kind, _ in sse_events(response.text)]
        self.assertEqual(kinds[-1], "done")

    async def test_managed_events_stream_as_job_and_replay_keeps_original_kinds(self):
        self.adapter.release.clear()
        response_task = asyncio.create_task(
            self.client.post(self.path + "/messages", json={"text": "run work"})
        )
        try:
            await asyncio.wait_for(self.adapter.entered.wait(), 2)
            handle = self.turns.current("w", "c")
            payloads = [
                {
                    "kind": "simulation.requested",
                    "jobId": "j_sim",
                    "experimentId": "e_one",
                },
                {
                    "kind": "job.running",
                    "jobId": "j_predict",
                    "jobKind": "timing_predict",
                    "status": "running",
                },
            ]
            for payload in payloads:
                self.store.turns.append_event(
                    handle.request.turn_id, payload["kind"], payload
                )
            handle.notify()
            self.adapter.release.set()
            response = await asyncio.wait_for(response_task, 3)
        finally:
            self.adapter.release.set()
            if not response_task.done():
                response_task.cancel()
            await asyncio.gather(response_task, return_exceptions=True)
        self.assertEqual(response.status_code, 200)
        events = sse_events(response.text)
        self.assertEqual(
            [payload for kind, payload in events if kind == "job"], payloads
        )
        replay = await self.client.get(
            self.path + f"/turns/{handle.request.turn_id}/replay"
        )
        self.assertEqual(replay.status_code, 200)
        for payload in payloads:
            self.assertIn(payload["kind"], replay.text)
        stored = self.store.turns.events("c", handle.request.turn_id)
        self.assertEqual(
            [
                event["payload"]
                for event in stored
                if event["kind"] in {p["kind"] for p in payloads}
            ],
            payloads,
        )

    async def test_two_http_turns_and_backwards_history(self):
        for index in range(2):
            response = await self.client.post(
                self.path + "/messages", json={"text": f"question {index}"}
            )
            self.assertEqual(response.status_code, 200)
            events = sse_events(response.text)
            self.assertEqual(events[-1][0], "done")
            self.assertEqual(events[-1][1]["outcome"], "final_answer")
            turn_id = response.headers["x-turn-id"]
            replay = await self.client.get(self.path + f"/turns/{turn_id}/replay")
            self.assertEqual(replay.status_code, 200)
            wrong = await self.client.get(
                self.path.replace("conversations/c", "conversations/other")
                + f"/turns/{turn_id}/replay"
            )
            self.assertEqual(wrong.status_code, 404)
        self.assertEqual(
            [request.session_id for request in self.adapter.requests], [None, "saved"]
        )
        index = (await self.client.get(self.path + "/turns")).json()["turns"]
        self.assertEqual(len(index), 2)
        self.assertTrue(
            all(
                turn["status"] == "complete" and turn["event_count"] > 0
                for turn in index
            )
        )
        history = (await self.client.get(self.path)).json()
        self.assertEqual(len(history["messages"]), 4)
        self.assertEqual(len({message["id"] for message in history["messages"]}), 4)
        page = (await self.client.get(self.path, params={"limit": 2})).json()
        self.assertEqual(page["message_page"]["start_index"], 2)
        earlier = (
            await self.client.get(self.path, params={"limit": 2, "before": 2})
        ).json()
        self.assertEqual(earlier["messages"], history["messages"][:2])
        self.assertEqual(
            (await self.client.get(self.path + "/stream")).status_code, 204
        )

    async def test_disconnect_reconnect_and_targeted_cancel(self):
        self.adapter.release.clear()
        connection = LiveRequest(
            self.app, self.path + "/messages", method="POST", body={"text": "work"}
        )
        self.addAsyncCleanup(connection.close)
        start = await connection.next()
        turn_id = dict(start["headers"])[b"x-turn-id"].decode()
        await asyncio.wait_for(self.adapter.entered.wait(), 2)
        await connection.close()
        self.assertIsNotNone(self.turns.current("w", "c"))
        self.assertEqual(
            (
                await self.client.post(
                    self.path + "/messages", json={"text": "duplicate"}
                )
            ).status_code,
            409,
        )
        stale = await self.client.post(self.path + "/cancel", params={"turn_id": "old"})
        self.assertEqual(stale.json(), {"cancelled": False, "stale": True})
        reconnect = asyncio.create_task(self.client.get(self.path + "/stream"))
        await asyncio.sleep(0)
        cancelled = await self.client.post(
            self.path + "/cancel", params={"turn_id": turn_id}
        )
        self.assertEqual(
            cancelled.json(), {"cancelled": True, "interrupted_role": "assistant"}
        )
        response = await reconnect
        self.assertEqual(response.headers["x-turn-id"], turn_id)
        self.assertEqual(sse_events(response.text)[-1][1]["outcome"], "cancelled")

    async def test_context_and_mode_options_survive_history(self):
        context = {
            "protocol": "vibesim.conversation-context/v2",
            "selection": None,
            "citationDictionary": dictionary().model_dump(by_alias=True),
        }
        response = await self.client.post(
            self.path + "/messages",
            json={
                "text": "inspect",
                "autonomous_mode": True,
                "analyzerContext": context,
            },
        )
        self.assertEqual(response.status_code, 200)
        done = sse_events(response.text)[-1][1]
        self.assertEqual(done["citation_dictionary_id"], "s_test:1")
        self.assertEqual(len(done["citations"]), 1)
        history = (await self.client.get(self.path)).json()
        self.assertEqual(history["messages"][0]["analyzer_context"], context)
        self.assertEqual(history["messages"][-1]["citations"], done["citations"])
        second = await self.client.post(
            self.path + "/messages",
            json={
                "text": "next",
                "agentMode": "orchestrated",
                "autonomous_mode": False,
            },
        )
        self.assertEqual(second.status_code, 200)
        self.assertTrue((await self.client.get(self.path)).json()["autonomous"])

    async def test_invalid_requests_do_not_write_messages(self):
        for payload in (
            {"text": " "},
            {"text": "x", "analyzerContext": {}},
            {"text": "x", "resumeRole": "implementer"},
        ):
            response = await self.client.post(self.path + "/messages", json=payload)
            self.assertIn(response.status_code, (400, 409, 422))
        self.assertEqual(self.store.conversations.messages("c"), ())
        self.assertEqual(
            (
                await self.client.get(self.path.replace("/w/", "/missing/") + "/stream")
            ).status_code,
            404,
        )

    async def test_machine_readable_mode_and_unavailable_errors(self):
        unknown = await self.client.post(
            self.path + "/messages", json={"text": "x", "agentMode": "other"}
        )
        self.assertEqual(unknown.status_code, 400)
        self.assertEqual(
            unknown.json()["detail"],
            {"code": "unknown_agent_mode", "agent_mode": "other"},
        )
        with patch.object(self.turns.driver.providers, "available", return_value=False):
            unavailable = await self.client.post(
                self.path + "/messages", json={"text": "x"}
            )
        self.assertEqual(unavailable.status_code, 409)
        self.assertEqual(
            unavailable.json()["detail"],
            {"code": "codex_family_unavailable", "families": ["test"]},
        )
        self.assertEqual(self.store.conversations.messages("c"), ())


class KeepaliveTests(unittest.IsolatedAsyncioTestCase):
    async def test_keepalive_fills_quiet_gaps_without_losing_or_reordering_frames(self):
        from vibesim_agent.api.conversations import KEEPALIVE_FRAME, with_keepalive

        async def frames():
            yield "a"
            await asyncio.sleep(0.12)
            yield "b"
            yield "c"

        out = [frame async for frame in with_keepalive(frames(), 0.05)]
        self.assertEqual([f for f in out if f != KEEPALIVE_FRAME], ["a", "b", "c"])
        self.assertEqual(out[0], "a")
        self.assertIn(KEEPALIVE_FRAME, out[1:-2])

    async def test_closing_the_keepalive_stream_closes_the_source(self):
        from contextlib import aclosing

        from vibesim_agent.api.conversations import with_keepalive

        closed = asyncio.Event()

        async def frames():
            try:
                yield "a"
                await asyncio.Event().wait()
            finally:
                closed.set()

        async with aclosing(with_keepalive(frames(), 0.05)) as kept:
            self.assertEqual(await anext(kept), "a")
            await anext(kept)  # a keepalive while the source waits
        self.assertTrue(closed.is_set())
