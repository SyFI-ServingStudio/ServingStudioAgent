import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import httpx
from pydantic import SecretStr, ValidationError

from vibesim_agent.prompts.render import Prompts
from vibesim_agent.services.name_generator import GeneratedNames, NameGenerator
from tests.test_prompt_bundle import legacy


class NameGeneratorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.prompts = Prompts.prepare(self.root / "prompts")
        self.golden = json.loads(
            (Path(__file__).parent / "fixtures/legacy_naming/golden.json").read_text()
        )

    def generator(self, factory, **overrides):
        return NameGenerator(
            **(
                {
                    "api_key": SecretStr("private-key"),
                    "model": "model",
                    "base_url": "https://naming.invalid/v1/",
                    "timeout": 2,
                    "prompts": self.prompts,
                    "client_factory": factory,
                }
                | overrides
            )
        )

    async def test_request_matches_legacy_schema_prompts_options_and_bounded_sources(
        self,
    ):
        requests = []
        response_data = self.golden["response"]

        def respond(request):
            requests.append(request)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps(response_data)}}]},
            )

        client_type = httpx.AsyncClient
        factory = lambda **kwargs: client_type(
            transport=httpx.MockTransport(respond), **kwargs
        )
        user, answer = "A" * 4500 + "middle" * 1000 + "Z" * 1500, "B" * 7000
        generator = self.generator(factory)
        result = await generator.generate(user, answer)
        self.assertEqual(len(requests), 1)
        self.assertEqual(result.model_dump(), self.golden["result"])
        self.assertEqual(result.workspace_name, "Workspace Name")
        # Legacy but for the product rename, which the prompt fixture defines.
        self.assertEqual(
            json.loads(legacy(requests[0].content.decode())),
            self.golden["request"]["body"],
        )
        self.assertEqual(
            str(requests[0].url), self.golden["request"]["url"]
        )
        self.assertEqual(
            requests[0].headers["Authorization"], self.golden["request"]["authorization"]
        )
        self.assertEqual(
            requests[0].extensions["timeout"], self.golden["request"]["timeout"]
        )
        body = json.loads(requests[0].content)
        self.assertEqual(body["max_tokens"], 96)
        self.assertTrue(body["provider"]["zdr"])
        self.assertNotIn("middle", body["messages"][1]["content"])

    async def test_disabled_configuration_does_not_create_client_or_read_prompts(self):
        factory = Mock()
        for secret in (None, SecretStr(""), SecretStr("   ")):
            generator = self.generator(
                factory, api_key=secret, prompts=Prompts(self.root / "absent")
            )
            self.assertFalse(generator.enabled)
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                await generator.generate("question", "answer")
        factory.assert_not_called()

    async def test_http_failure_closes_client_without_logging_credentials_or_payload(
        self,
    ):
        clients = []

        def factory(**kwargs):
            client = httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(503, text="private response")
                ),
                **kwargs,
            )
            clients.append(client)
            return client

        with (
            self.assertNoLogs("vibesim_agent", level="DEBUG"),
            self.assertRaises(httpx.HTTPStatusError),
        ):
            await self.generator(factory).generate("private prompt", "private answer")
        self.assertTrue(clients[0].is_closed)

    async def test_total_timeout_bounds_nonreturning_transport_and_closes_client(self):
        clients = []
        cancelled = asyncio.Event()

        async def hanging(request):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        def factory(**kwargs):
            client = httpx.AsyncClient(transport=httpx.MockTransport(hanging), **kwargs)
            clients.append(client)
            return client

        safety_deadline = asyncio.timeout(1)
        with self.assertRaises(TimeoutError):
            async with safety_deadline:
                await self.generator(factory, timeout=0.01).generate(
                    "question", "answer"
                )
        self.assertFalse(safety_deadline.expired())
        self.assertTrue(cancelled.is_set())
        self.assertTrue(clients[0].is_closed)

    async def test_invalid_content_and_names_are_rejected(self):
        for content in (
            None,
            "not-json",
            '{"workspace_name":"ab","conversation_title":"Title"}',
            '{"workspace_name":"   ","conversation_title":"Title"}',
        ):
            with self.subTest(content=content):

                def factory(content=content, **kwargs):
                    return httpx.AsyncClient(
                        transport=httpx.MockTransport(
                            lambda request: httpx.Response(
                                200,
                                json={"choices": [{"message": {"content": content}}]},
                            )
                        ),
                        **kwargs,
                    )

                with self.assertRaises(ValueError):
                    await self.generator(factory).generate("question", "answer")

    def test_generated_name_validation_preserves_legacy_multiline_and_length_contract(
        self,
    ):
        for case in self.golden["validation_cases"]:
            payload = case["input"]
            with self.subTest(payload=payload):
                if case["validation_error"]:
                    with self.assertRaises(ValidationError):
                        GeneratedNames.model_validate(payload)
                else:
                    self.assertEqual(
                        GeneratedNames.model_validate(payload).model_dump(),
                        case["result"],
                    )


class BranchTopicTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.prompts = Prompts.prepare(self.root / "prompts")

    def generator(self, respond, timeout=8.0):
        self.requests = []

        def record(request):
            self.requests.append(request)
            return respond(request)

        client_type = httpx.AsyncClient
        return NameGenerator(
            api_key=SecretStr("key"),
            model="naming-model",
            base_url="https://naming.invalid/api/v1",
            timeout=timeout,
            prompts=self.prompts,
            client_factory=lambda **kwargs: client_type(
                transport=httpx.MockTransport(record), **kwargs
            ),
        )

    @staticmethod
    def reply(content):
        return lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    async def test_asks_with_the_branch_contract_and_returns_the_branch(self):
        generator = self.generator(self.reply(json.dumps({"branch": " glm5-decode-profile "})))
        self.assertEqual(await generator.branch_topic("为什么 GLM5.2 decode 这么慢"), "glm5-decode-profile")
        body = json.loads(self.requests[0].content)
        self.assertEqual(
            body["messages"][0]["content"],
            (self.prompts.directory / "branch-system.txt").read_text().strip(),
        )
        # The request is passed through unescaped, so a Chinese question reaches
        # the model as Chinese rather than as `\\u` escapes.
        self.assertIn("为什么 GLM5.2 decode 这么慢", body["messages"][1]["content"])
        self.assertEqual(
            body["response_format"]["json_schema"]["schema"]["required"], ["branch"]
        )
        # Without this, providers that reason first spend the whole budget on it.
        self.assertEqual(body["reasoning"], {"enabled": False})

    async def test_the_wait_is_bounded_well_below_the_naming_timeout(self):
        # The reader is waiting on "Send" while this runs.
        seen = []
        client_type = httpx.AsyncClient
        generator = NameGenerator(
            api_key=SecretStr("key"),
            model="m",
            base_url="https://naming.invalid",
            timeout=30.0,
            prompts=self.prompts,
            client_factory=lambda **kwargs: seen.append(kwargs["timeout"])
            or client_type(
                transport=httpx.MockTransport(self.reply(json.dumps({"branch": "x-y"}))),
                **kwargs,
            ),
        )
        await generator.branch_topic("q")
        self.assertEqual(seen, [4.0])

    async def test_an_answer_without_a_branch_is_an_error_for_the_caller(self):
        generator = self.generator(self.reply(json.dumps({"branch": "  "})))
        with self.assertRaises(ValueError):
            await generator.branch_topic("q")
