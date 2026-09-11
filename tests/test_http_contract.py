"""Acceptance scenarios shared by the legacy and replacement compositions."""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from contract_support import legacy_application

from tests.http_support import LiveRequest, sse_events

ROOT = "/api/agent/v1"
AUTH = {"Authorization": "Bearer contract-token"}


class HttpContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = self.enterContext(TemporaryDirectory())
        self.calls = []
        self.replies = []
        self.scripted = False
        self.gate = None

        async def runner(container, prompt, **options):
            self.calls.append({"prompt": prompt, **options})
            role = options["label"]
            yield {"kind": "role_start", "role": role}
            yield {
                "kind": "session",
                "role": role,
                "session_id": options.get("session_id") or f"session-{role}",
            }
            yield {"kind": "role_ready", "role": role}
            if self.gate is not None:
                await self.gate.wait()
            if self.scripted and not self.replies:
                raise AssertionError("Unexpected CLI call after script exhausted")
            reply = (
                self.replies.pop(0)
                if self.replies
                else {
                    "action": "final_answer",
                    "message": "Contract answer.",
                    "task": "",
                }
            )
            if isinstance(reply, Exception):
                raise reply
            yield {"kind": "final", "text": json.dumps(reply)}

        app = await self.enterAsyncContext(
            legacy_application(Path(self.directory), runner)
        )
        self.app = app
        self.client = await self.enterAsyncContext(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            )
        )

    async def conversation(self, *, mode="single", autonomous=False, tools=False):
        path = f"{ROOT}{'/tools' if tools else ''}/workspaces/w_main/conversations"
        response = await self.client.post(
            path, headers=AUTH, json={"agent_mode": mode, "autonomous": autonomous}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return f"{path}/{response.json()['id']}"

    async def test_both_modes_preserve_message_identity_and_resume(self):
        for mode, role in (("single", "assistant"), ("orchestrated", "orchestrator")):
            with self.subTest(mode=mode):
                path = await self.conversation(mode=mode)
                first = await self.client.post(
                    f"{path}/messages", json={"text": "first"}
                )
                self.assertEqual(first.status_code, 200, first.text)
                turn_id = first.headers["x-turn-id"]
                self.assertEqual(sse_events(first.text)[-1][0], "done")
                history = (await self.client.get(path)).json()["messages"]
                self.assertEqual([m["role"] for m in history], ["user", "assistant"])
                self.assertEqual({m["turn_id"] for m in history}, {turn_id})
                identities = [m["id"] for m in history]
                self.assertLess(identities[0], identities[1])
                second = await self.client.post(
                    f"{path}/messages", json={"text": "second"}
                )
                self.assertEqual(second.status_code, 200)
                self.assertEqual(self.calls[-1]["label"], role)
                self.assertEqual(self.calls[-1]["session_id"], f"session-{role}")
                restored = (await self.client.get(path)).json()["messages"]
                self.assertEqual(restored[:2], history)
                self.assertNotEqual(restored[-1]["turn_id"], turn_id)
                self.assertEqual(
                    (await self.client.get(f"{path}/stream")).status_code, 204
                )

    async def test_delegation_runs_real_orchestration_and_preserves_each_session(self):
        path = await self.conversation(mode="orchestrated")
        self.scripted = True
        self.replies = [
            {"action": "delegate", "message": "", "task": "Read a file."},
            {"action": "final_answer", "message": "Read the file."},
            {"action": "final_answer", "message": "Reviewed.", "task": ""},
        ]
        response = await self.client.post(f"{path}/messages", json={"text": "inspect"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [c["label"] for c in self.calls],
            ["orchestrator", "implementer", "orchestrator"],
        )
        self.assertEqual(self.calls[-1]["session_id"], "session-orchestrator")
        self.assertEqual(sse_events(response.text)[-1][1]["text"], "Reviewed.")
        self.replies = [
            {"action": "delegate", "message": "", "task": "Read another file."},
            {"action": "final_answer", "message": "Read another file."},
            {"action": "final_answer", "message": "Reviewed again.", "task": ""},
        ]
        followup = await self.client.post(f"{path}/messages", json={"text": "another"})
        self.assertEqual(sse_events(followup.text)[-1][1]["text"], "Reviewed again.")
        self.assertEqual(self.calls[4]["label"], "implementer")
        self.assertEqual(self.calls[4]["session_id"], "session-implementer")
        self.assertEqual(len(self.calls), 6)

    async def test_authentication_and_public_skill(self):
        self.assertEqual(
            (await self.client.get(f"{ROOT}/tools/workspaces")).status_code, 401
        )
        wrong = await self.client.get(
            f"{ROOT}/tools/workspaces", headers={"Authorization": "Bearer wrong"}
        )
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(
            (
                await self.client.get(f"{ROOT}/tools/workspaces", headers=AUTH)
            ).status_code,
            200,
        )
        self.assertEqual(
            (await self.client.get(f"{ROOT}/tools/skill")).status_code, 200
        )
        self.assertEqual((await self.client.get(f"{ROOT}/workspaces")).status_code, 200)
        denied = await self.client.post(
            f"{ROOT}/internal/managed-jobs/register",
            json={"jobKind": "timing_predict", "artifactRoot": "logs/example"},
        )
        self.assertIn(denied.status_code, (401, 403))
        api_token_is_not_capability = await self.client.post(
            f"{ROOT}/internal/managed-jobs/register",
            headers=AUTH,
            json={"jobKind": "timing_predict", "artifactRoot": "logs/example"},
        )
        self.assertIn(api_token_is_not_capability.status_code, (401, 403))

    async def test_sync_and_eval_interpret_question_as_question(self):
        question = {
            "action": "request_user_input",
            "message": "Which model?",
            "task": "",
        }
        path = await self.conversation(tools=True)
        self.replies = [question, question]
        sync = await self.client.post(
            f"{path}/messages", headers=AUTH, json={"text": "help"}
        )
        evaluation = await self.client.post(
            f"{ROOT}/tools/eval",
            headers=AUTH,
            json={"prompt": "help", "agent_mode": "single"},
        )
        for response in (sync, evaluation):
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["ok"])
            self.assertEqual(response.json()["outcome"], "request_user_input")
            self.assertEqual(response.json()["final"], "Which model?")

    async def test_replay_is_repeatable_and_rejects_another_conversation(self):
        path = await self.conversation()
        sent = await self.client.post(f"{path}/messages", json={"text": "hello"})
        turn_id = sent.headers["x-turn-id"]
        before = (await self.client.get(path)).json()
        turns = (await self.client.get(f"{path}/turns")).json()["turns"]
        self.assertEqual([t["turn_id"] for t in turns], [turn_id])
        replay = await self.client.get(f"{path}/turns/{turn_id}/replay")
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(len(sse_events(replay.text)), turns[0]["event_count"])
        again = await self.client.get(f"{path}/turns/{turn_id}/replay")
        self.assertEqual(again.content, replay.content)
        self.assertEqual((await self.client.get(path)).json(), before)
        other = await self.conversation()
        self.assertEqual(
            (await self.client.get(f"{other}/turns/{turn_id}/replay")).status_code, 404
        )

    async def test_invalid_input_does_not_append_a_message(self):
        path = await self.conversation()
        for body in ({"text": "  "}, {"text": "hello", "agent_mode": "unknown"}):
            response = await self.client.post(f"{path}/messages", json=body)
            self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual((await self.client.get(path)).json()["messages"], [])
        self.assertEqual(self.calls, [])

    async def test_disconnect_reconnect_and_targeted_stop_keep_conversation_usable(
        self,
    ):
        path = await self.conversation()
        self.gate = asyncio.Event()
        connection = LiveRequest(
            self.app, f"{path}/messages", method="POST", body={"text": "wait"}
        )
        self.addAsyncCleanup(connection.close)
        start = await connection.next()
        self.assertEqual(start["status"], 200)
        turn_id = dict(start["headers"])[b"x-turn-id"].decode()
        initial_bodies = []
        while True:
            frame = await connection.next()
            initial_bodies.append(frame.get("body", b""))
            if b"event: role_ready" in frame.get("body", b""):
                break
        await connection.close()
        conflict = await self.client.post(f"{path}/messages", json={"text": "too soon"})
        self.assertEqual(conflict.status_code, 409)
        stale = await self.client.post(f"{path}/cancel", params={"turn_id": "previous"})
        self.assertEqual(stale.json(), {"cancelled": False, "stale": True})
        resumed = LiveRequest(self.app, f"{path}/stream")
        self.addAsyncCleanup(resumed.close)
        self.assertEqual(
            dict((await resumed.next())["headers"])[b"x-turn-id"].decode(), turn_id
        )
        stopped = await self.client.post(f"{path}/cancel", params={"turn_id": turn_id})
        self.assertEqual(
            stopped.json(), {"cancelled": True, "interrupted_role": "assistant"}
        )
        bodies = []
        while True:
            frame = await resumed.next()
            bodies.append(frame.get("body", b""))
            if not frame.get("more_body", False):
                break
        events = sse_events(b"".join(bodies).decode())
        initial_events = sse_events(b"".join(initial_bodies).decode())
        self.assertEqual(events[: len(initial_events)], initial_events)
        ending = events[-1]
        self.assertEqual(ending[0], "done")
        self.assertEqual(ending[1]["outcome"], "cancelled")
        history = (await self.client.get(path)).json()
        self.assertEqual(history["interrupted_role"], "assistant")
        self.assertEqual(
            history["messages"][-1]["activity"][-1]["outcome"], "cancelled"
        )
        self.gate = None
        next_turn = await self.client.post(
            f"{path}/messages", json={"text": "continue"}
        )
        self.assertEqual(next_turn.status_code, 200)
        self.assertEqual(self.calls[-1]["session_id"], "session-assistant")
