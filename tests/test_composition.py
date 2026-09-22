import asyncio
import json
import logging
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests import test_builtin_providers as fixtures
from tests.runtime_fixtures import docker_execution
from vibesim_agent.composition import build_builtin_setup
from vibesim_agent.domain.roles import Role
from vibesim_agent.prompts.render import Prompts
from vibesim_agent.providers.base import AgentRequest
from vibesim_agent.providers.claude.adapter import ClaudeAdapter
from vibesim_agent.providers.codex.adapter import CodexAdapter
from vibesim_agent.runtime.command import ExecutionEnvironment
from vibesim_agent.runtime.invocation import InvocationHome, RoleContext


class CompositionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        fixture = fixtures.ConfiguredProviderTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        deepseek_home = self.root / ".deepseek"
        deepseek_home.mkdir()
        (deepseek_home / "config.toml").write_text('model_provider="openai"\n')
        (deepseek_home / "auth.json").write_text('{"token":"fixture"}')
        fixture.document["providers"]["deepseek"] = {
            "adapter": "codex",
            "home": str(deepseek_home),
            "environment": {"VLLM_API_KEY": "VLLM_API_KEY"},
            "default_model": "deepseek-v3",
            "default_effort": "high",
            "models": {"deepseek-v3": {"efforts": ["high"]}},
        }
        fixture.document["providers"]["claude"]["environment"] = {
            "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY"
        }
        fixture.document["defaults"] = {
            "orchestrator": "gpt",
            "implementer": "gpt",
            "assistant": "gpt",
        }
        (self.root / "providers.yaml").write_text(json.dumps(fixture.document))
        self.settings = fixture.settings(
            VLLM_API_KEY="selected-deepseek",
            ANTHROPIC_API_KEY="selected-claude",
        )
        self.environment = ExecutionEnvironment(
            self.settings.container, self.settings.agent, "lock", "/context.json"
        )
        self.prompts = Prompts.prepare(self.root / "prompts")

        def home(request):
            path = (
                self.root
                / "runtime"
                / request.selection.provider_id
                / request.role.value
            )
            return InvocationHome(path, str(path))

        self.home = home
        self.setup = build_builtin_setup(
            self.settings,
            home,
            self.environment,
            self.prompts,
            host_environment={
                "PATH": "/usr/bin",
                "DOCKER_HOST": "unix:///test.sock",
                "ANTHROPIC_AUTH_TOKEN": "unselected-host-secret",
            },
        )

    def request(self, provider_id):
        return AgentRequest(
            "w",
            "c",
            "t",
            Role.ASSISTANT,
            "question",
            docker_execution(self.environment),
            self.setup.registry.select(provider_id),
            output_schema=Path("/contracts/assistant.schema.json"),
        )

    def test_real_adapter_types_defaults_and_isolated_credentials(self):
        gpt = self.setup.registry.provider("gpt").adapter
        deepseek = self.setup.registry.provider("deepseek").adapter
        claude = self.setup.registry.provider("claude").adapter
        self.assertIsInstance(gpt, CodexAdapter)
        self.assertIsInstance(deepseek, CodexAdapter)
        self.assertIsNot(gpt, deepseek)
        self.assertIsInstance(claude, ClaudeAdapter)
        self.assertEqual({r.provider_id for r in self.setup.defaults.values()}, {"gpt"})
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", claude.process_environment)
        self.assertNotIn("ANTHROPIC_API_KEY", deepseek.process_environment)
        self.assertNotIn("VLLM_API_KEY", gpt.process_environment)
        self.assertEqual(
            deepseek.process_environment["VLLM_API_KEY"], "selected-deepseek"
        )
        self.assertEqual(
            claude.process_environment["ANTHROPIC_API_KEY"], "selected-claude"
        )
        self.assertNotIn("sonnet", self.setup.model_aliases)
        self.assertEqual(self.setup.docker_environment, {
            "PATH": "/usr/bin", "DOCKER_HOST": "unix:///test.sock",
        })
        for adapter in (gpt, deepseek, claude):
            for key, value in self.setup.docker_environment.items():
                self.assertEqual(adapter.process_environment[key], value)

    def test_copied_catalog_drives_each_home_command_without_shared_mutation(self):
        request = self.request("gpt")
        home = self.home(request)
        profile = self.settings.providers["gpt"].home
        (profile / "models_catalog.json").write_text("{}")
        self.setup.runtimes["gpt"].prepare(home, RoleContext(skills="/workspace/skills"))
        adapter = self.setup.registry.provider("gpt").adapter
        first = adapter.command_for_home(home)
        self.assertEqual(first.catalog_filename, "models_catalog.json")
        (profile / "models_catalog.json").unlink()
        self.setup.runtimes["gpt"].prepare(home, RoleContext(skills="/workspace/skills"))
        second = adapter.command_for_home(home)
        self.assertIsNone(second.catalog_filename)
        self.assertIsNone(adapter.command.catalog_filename)
        self.assertEqual(first.catalog_filename, "models_catalog.json")

    def test_guard_is_used_by_actual_profile_supply(self):
        profile = self.settings.providers["gpt"].home
        (profile / "config.toml").write_text('model_provider="changed"\n')
        home = self.home(self.request("gpt"))
        with self.assertRaisesRegex(ValueError, "backend changed"):
            self.setup.runtimes["gpt"].prepare(home, RoleContext(skills="/workspace/skills"))
        self.assertFalse(home.host.exists())

    async def test_adapter_processes_receive_selected_environment_and_resume(self):
        original = asyncio.create_subprocess_exec
        captured = []

        async def spawn(*arguments, **kwargs):
            captured.append((arguments, kwargs["env"]))
            if "vibesim-claude" in arguments:
                records = [
                    {
                        "type": "result",
                        "subtype": "success",
                        "session_id": "saved",
                        "structured_output": {
                            "action": "final_answer",
                            "message": "Done",
                        },
                    }
                ]
            else:
                records = [
                    {"type": "thread.started", "thread_id": "saved"},
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "Done",
                            "phase": "final_answer",
                        },
                    },
                ]
            code = (
                "import sys; sys.stdin.read(); print("
                + repr("\n".join(json.dumps(record) for record in records))
                + ")"
            )
            return await original(sys.executable, "-c", code, **kwargs)

        for provider_id in ("gpt", "deepseek", "claude"):
            request = self.request(provider_id)
            self.setup.runtimes[provider_id].prepare(self.home(request), RoleContext(skills="/workspace/skills"))
            with patch("asyncio.create_subprocess_exec", spawn):
                for session in (None, "saved"):
                    events = [
                        event
                        async for event in self.setup.registry.run(
                            replace(request, session_id=session)
                        )
                    ]
                    self.assertNotIn("failure", events[-1])
                    self.assertEqual(events[-1]["kind"], "final")
            args, environment = captured[-1]
            self.assertEqual(environment["DOCKER_HOST"], "unix:///test.sock")
            self.assertFalse(
                any(
                    "selected-" in arg or "unselected-host-secret" in arg
                    for arg in args
                )
            )
            if provider_id == "deepseek":
                self.assertIn("VLLM_API_KEY", args)
                self.assertEqual(environment["VLLM_API_KEY"], "selected-deepseek")
                self.assertIn("--output-schema", args)
            if provider_id == "claude":
                self.assertIn("ANTHROPIC_API_KEY", args)
                self.assertIn("--resume", args)

    async def test_stop_helpers_use_same_docker_environment(self):
        """Each provider signals with its own credential-bearing environment.

        `docker exec -e NAME` inherits by name from the client process, so the
        stop helper has to run with the same environment as the call it stops.
        """
        original = asyncio.create_subprocess_exec
        environments = []

        async def spawn(*args, **kwargs):
            environments.append(kwargs["env"])
            return await original(sys.executable, "-c", "pass", **kwargs)

        for provider_id in ("gpt", "claude"):
            adapter = self.setup.registry.provider(provider_id).adapter
            invocation = docker_execution(self.environment).invocation(
                execution_id="e",
                home=InvocationHome(Path("/tmp"), "/tmp"),
                logger=logging.getLogger("composition-test"),
                label=provider_id,
                process_environment=adapter.process_environment,
            )
            with patch("asyncio.create_subprocess_exec", spawn):
                await invocation.signal("container", "/call.pid", "INT")
            self.assertEqual(environments[-1], adapter.process_environment)
