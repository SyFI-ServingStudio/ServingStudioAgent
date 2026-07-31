from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import httpx

from backend import naming


class FakeAsyncClient:
    def __init__(self, response: httpx.Response, captured: dict) -> None:
        self.response = response
        self.captured = captured

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, url: str, *, headers: dict, json: dict) -> httpx.Response:
        self.captured.update({"url": url, "headers": headers, "json": json})
        return self.response


class OpenRouterNamingTest(unittest.IsolatedAsyncioTestCase):
    async def test_uses_structured_zdr_request_and_bounds_source_text(self) -> None:
        generated = {
            "workspace_name": "Llama 3 H200 Capacity",
            "conversation_title": "Find Maximum SLO Goodput",
        }
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {"message": {"content": json.dumps(generated)}}
                ]
            },
        )
        captured: dict = {}
        client = FakeAsyncClient(response, captured)
        with (
            patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": "secret",
                    "VIBESIM_NAMING_MODEL": "deepseek/deepseek-v4-flash",
                },
                clear=False,
            ),
            patch.object(naming.httpx, "AsyncClient", return_value=client),
        ):
            names = await naming.generate_names("x" * 7000, "final answer")

        self.assertEqual(names.workspace_name, generated["workspace_name"])
        self.assertEqual(
            captured["url"], "https://openrouter.ai/api/v1/chat/completions"
        )
        self.assertEqual(captured["json"]["response_format"]["type"], "json_schema")
        self.assertTrue(captured["json"]["response_format"]["json_schema"]["strict"])
        self.assertEqual(
            captured["json"]["provider"],
            {
                "require_parameters": True,
                "zdr": True,
                "allow_fallbacks": True,
            },
        )
        prompt = captured["json"]["messages"][1]["content"]
        self.assertLess(len(prompt), 6500)
        self.assertNotIn("secret", prompt)

    def test_schedule_is_disabled_without_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                naming.schedule_auto_naming(
                    unittest.mock.Mock(),
                    "w_test",
                    "conversation",
                    "question",
                    "answer",
                )
            )

    def test_accepts_established_openroute_key_alias(self) -> None:
        with patch.dict(os.environ, {"OPENROUTE_KEY": "secret"}, clear=True):
            config = naming._naming_config()

        self.assertIsNotNone(config)
        self.assertEqual(config[0], "secret")

    def test_standard_openrouter_key_takes_precedence(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "standard",
                "OPENROUTE_KEY": "alias",
            },
            clear=True,
        ):
            config = naming._naming_config()

        self.assertIsNotNone(config)
        self.assertEqual(config[0], "standard")


if __name__ == "__main__":
    unittest.main()
