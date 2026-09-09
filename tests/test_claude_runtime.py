import asyncio
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from test_turn import _run

from backend import app as app_module
from backend.codex_runtime import agent_cli, claude_cli, config, docker
from backend.codex_runtime.claude_command import (
    build_claude_exec_command,
    output_schema,
)
from backend.codex_runtime.claude_events import ClaudeOutputCollector
from backend.codex_runtime.exec_types import CodexExecRequest
from backend.store import Store, WorkspaceRegistry


def request(**overrides):
    values = {
        "container": "test-container",
        "prompt": "question",
        "label": "assistant",
        "workspace_id": "w_main",
        "conversation_id": "c_test",
        "turn_id": "t_test",
        "model_id": "sonnet",
        "effort": "high",
        "output_schema": config.ASSISTANT_SCHEMA_IN_CONTAINER,
    }
    values.update(overrides)
    return CodexExecRequest(**values)


def result(**overrides):
    value = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "session_id": "claude-session",
        "result": "Do not use this raw fallback",
        "structured_output": {"action": "final_answer", "message": "Done"},
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 30,
            "output_tokens": 40,
        },
    }
    value.update(overrides)
    return value


def consume(collector, event):
    return collector.events_from_stdout_line(json.dumps(event).encode())


class ClaudeProtocolTests(unittest.TestCase):
    def test_cli_schema_omits_unsupported_dialect_without_changing_contract(self):
        for role in ("assistant", "orchestrator", "implementer"):
            with self.subTest(role=role):
                call = request(output_schema=f"{role}.schema.json")
                original = output_schema(call)
                command = build_claude_exec_command(call)
                cli_schema = json.loads(command[command.index("--json-schema") + 1])
                self.assertNotIn("$schema", cli_schema)
                self.assertEqual(
                    cli_schema, {k: v for k, v in original.items() if k != "$schema"}
                )
                self.assertEqual(output_schema(call), original)

    def test_launch_wrapper_records_pid_and_preserves_cli_arguments(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_cli = root / "claude"
            fake_cli.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake_cli.chmod(0o700)
            pid_path = root / "active.pid"
            command = build_claude_exec_command(request())
            local_command = command[command.index("test-container") + 1 :]
            completed = subprocess.run(
                local_command,
                input="",
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "PATH": f"{root}:{os.environ['PATH']}",
                    "VIBESIM_AGENT_PID_FILE": str(pid_path),
                },
                check=True,
            )
            self.assertTrue(pid_path.read_text().strip().isdecimal())
            arguments = completed.stdout.splitlines()
            self.assertEqual(arguments[0], "-p")
            self.assertEqual(arguments[arguments.index("--model") + 1], "sonnet")
            self.assertIn("--json-schema", arguments)

    def test_resume_command_isolates_state_and_explicitly_configures_contract(self):
        command = build_claude_exec_command(request(session_id="saved-session"))
        self.assertEqual(command[command.index("--resume") + 1], "saved-session")
        self.assertIn(
            "CLAUDE_CONFIG_DIR="
            + config.role_codex_home_in_container("assistant")
            + "/claude",
            command,
        )
        self.assertEqual(
            command[command.index("--append-system-prompt-file") + 1],
            "/workspace/AGENTS.md",
        )
        self.assertIn("--strict-mcp-config", command)
        self.assertIn(
            "analyzer",
            json.loads(command[command.index("--mcp-config") + 1])["mcpServers"],
        )
        self.assertNotIn("--resume", build_claude_exec_command(request()))
        self.assertNotIn(
            "--bare", command
        )  # OAuth and the managed skills remain usable.
        self.assertNotIn("service_tier", " ".join(command))
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "never-in-command"}):
            self.assertNotIn(
                "never-in-command", " ".join(build_claude_exec_command(request()))
            )

    def test_init_does_not_unlock_interrupt_and_only_final_result_routes(self):
        collector = ClaudeOutputCollector(request())
        events = consume(
            collector,
            {"type": "system", "subtype": "init", "session_id": "claude-session"},
        )
        self.assertEqual([e["kind"] for e in events], ["session"])
        events = consume(
            collector,
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": '{"action":"delegate","task":"wrong"}',
                        },
                        {
                            "type": "text",
                            "text": '{"action":"progress","message":"Reading"}',
                        },
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "README.md"},
                        },
                    ]
                },
            },
        )
        self.assertEqual(
            [e["kind"] for e in events],
            ["role_ready", "intermediate_output", "tool_call"],
        )
        self.assertIsNone(collector.result)
        consume(collector, result())
        events = collector.finish(0, 100)
        self.assertEqual(events[0]["tokens"], {"read": 30, "prefill": 30, "output": 40})
        self.assertEqual(json.loads(events[-1]["text"])["message"], "Done")

    def test_error_exit_or_invalid_schema_never_becomes_success(self):
        for payload, exitcode in [
            (result(is_error=True), 0),
            (result(), 1),
            (result(structured_output=None), 0),
            (result(structured_output={"action": "delegate", "message": "bad"}), 0),
            (None, 0),
        ]:
            with self.subTest(payload=payload, exitcode=exitcode):
                collector = ClaudeOutputCollector(request())
                if payload:
                    consume(collector, payload)
                final = collector.finish(exitcode, 5)[-1]
                self.assertIn("failure", final)
                self.assertEqual(final["text"], "")

    def test_nested_messages_do_not_replace_session_or_result(self):
        collector = ClaudeOutputCollector(request(session_id="owner"))
        self.assertEqual(consume(collector, result(parent_tool_use_id="nested")), [])
        self.assertEqual(collector.session_id, "owner")
        self.assertIsNone(collector.result)

    def test_partial_and_malformed_messages_are_not_decisions(self):
        collector = ClaudeOutputCollector(request())
        for line in (b"not-json", b"null", b"[]", b'{"type":"assistant","message":42}'):
            collector.events_from_stdout_line(line)
        consume(
            collector,
            {"type": "stream_event", "event": {"type": "content_block_delta"}},
        )
        self.assertIsNone(collector.result)


class ClaudeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_and_mixed_role_dispatch_and_resume(self):
        calls = []

        async def fake_codex(container, prompt, **kwargs):
            calls.append(("codex", kwargs))
            text = (
                {"action": "delegate", "message": "", "task": "Inspect README"}
                if len(calls) == 1
                else {"action": "final_answer", "message": "Reviewed", "task": ""}
            )
            yield {"kind": "final", "text": json.dumps(text)}

        async def fake_claude(container, prompt, **kwargs):
            calls.append(("claude", kwargs))
            yield {
                "kind": "session",
                "role": kwargs["label"],
                "session_id": "saved-claude",
            }
            yield {
                "kind": "final",
                "text": '{"action":"final_answer","message":"Read README"}',
            }

        with (
            patch.object(agent_cli, "run_codex", fake_codex),
            patch.object(agent_cli, "run_claude", fake_claude),
        ):
            events = await _run(
                agent_cli.run_agent,
                role_runtimes={
                    "orchestrator": {"model": "gpt-5.6-sol"},
                    "implementer": {"model": "sonnet"},
                },
                sessions={"implementer": "prior-claude"},
            )
            self.assertEqual([c[0] for c in calls], ["codex", "claude", "codex"])
            self.assertEqual(calls[1][1]["session_id"], "prior-claude")
            self.assertEqual(events[-1]["text"], "Reviewed")
            calls.clear()
            events = await _run(
                agent_cli.run_agent,
                agent_mode="single",
                role_runtimes={"assistant": {"model": "sonnet"}},
            )
            self.assertEqual([c[0] for c in calls], ["claude"])
            self.assertEqual(events[-1]["text"], "Read README")

    async def test_real_subprocess_reads_fragmented_stream_and_reports_usage(self):
        payload = json.dumps(result())
        # Exercise actual pipes and EOF, including a result larger than the
        # subprocess reader's default line limit and without a trailing newline.
        payload = payload.replace("Do not use this raw fallback", "x" * 100000)
        script = "import sys; sys.stdin.read(); sys.stdout.write(" + repr(payload) + ")"
        with patch.object(
            claude_cli,
            "build_claude_exec_command",
            return_value=[sys.executable, "-c", script],
        ):
            events = [
                e
                async for e in claude_cli.run_claude(
                    "unused",
                    "hello",
                    label="assistant",
                    workspace_id="w_main",
                    conversation_id="test",
                    turn_id="test",
                    model_id="sonnet",
                    effort="high",
                    output_schema=config.ASSISTANT_SCHEMA_IN_CONTAINER,
                )
            ]
        self.assertEqual(events[0]["kind"], "role_start")
        self.assertEqual(events[-2]["tokens"]["prefill"], 30)
        self.assertEqual(json.loads(events[-1]["text"])["message"], "Done")

    async def test_idle_timeout_stops_container_agent_and_does_not_repair(self):
        actual_spawn = asyncio.create_subprocess_exec
        processes = []

        async def spawn(*args, **kwargs):
            process = await actual_spawn(
                sys.executable,
                "-c",
                "import sys,time; sys.stdin.read(); time.sleep(30)",
                **kwargs,
            )
            processes.append(process)
            return process

        async def signal(container, name, path):
            self.assertEqual(name, "INT")
            processes[0].terminate()

        with (
            patch.object(claude_cli.asyncio, "create_subprocess_exec", spawn),
            patch.object(claude_cli, "signal_claude", signal),
            patch.object(claude_cli, "CODEX_IDLE_TIMEOUT", 0.05),
        ):
            events = [
                e
                async for e in claude_cli.run_claude(
                    "container",
                    "hello",
                    label="assistant",
                    workspace_id="w_main",
                    conversation_id="test",
                    turn_id="test",
                    model_id="sonnet",
                    effort="high",
                )
            ]
        self.assertEqual(events[-1]["failure"]["code"], "agent_call_timeout")
        self.assertIsNotNone(processes[0].returncode)

    async def test_cancel_signals_agent_not_only_docker_exec(self):
        actual_spawn = asyncio.create_subprocess_exec
        processes = []
        running = asyncio.Event()

        async def spawn(*args, **kwargs):
            process = await actual_spawn(
                sys.executable, "-c", "import time; time.sleep(30)", **kwargs
            )
            processes.append(process)
            return process

        async def signal(container, name, path):
            processes[0].terminate()

        async def collect():
            async for event in claude_cli.run_claude(
                "container",
                "hello",
                label="assistant",
                workspace_id="w_main",
                conversation_id="test",
                turn_id="test",
                model_id="sonnet",
                effort="high",
            ):
                if event["kind"] == "role_start":
                    running.set()

        with (
            patch.object(claude_cli.asyncio, "create_subprocess_exec", spawn),
            patch.object(
                claude_cli, "signal_claude", AsyncMock(side_effect=signal)
            ) as signal_mock,
        ):
            task = asyncio.create_task(collect())
            await running.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(signal_mock.await_args.args[:2], ("container", "INT"))
        self.assertIsNotNone(processes[0].returncode)


class ClaudeStorageTests(unittest.TestCase):
    def test_single_claude_container_setup_needs_no_codex_credentials(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "repo"
            workspace.mkdir()
            (workspace / "AGENTS.md").touch()
            with (
                patch.object(config, "AGENT_WORKSPACES_ROOT", root / "state"),
                patch.dict(
                    os.environ, {"ANTHROPIC_API_KEY": "never-log-me"}, clear=True
                ),
                patch.object(docker, "container_running", return_value=False),
                patch.object(docker, "_submodule_mount_args", return_value=[]),
                patch.object(docker, "HOST_HF_HOME", None),
                patch.object(docker, "CODEX_DOCKER_GPUS", ""),
                patch.object(docker.subprocess, "run"),
                patch.object(docker, "run_checked") as execute,
            ):
                docker.ensure_container(
                    "w",
                    "c",
                    workspace,
                    "workspace-write",
                    agent_mode="single",
                    role_families={"assistant": "claude"},
                )
            command = execute.call_args_list[0].args[0]
            self.assertIn("--init", command)
            self.assertIn("ANTHROPIC_API_KEY", command)
            self.assertNotIn("never-log-me", " ".join(command))
            self.assertIn("command -v claude", execute.call_args_list[1].args[0][-1])
            self.assertTrue((root / "state/w/codex/c/assistant/claude").is_dir())

    def test_catalog_does_not_require_codex_config_and_hides_unsupported_tiers(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(config.codex_family("claude").available)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=True):
            models = app_module.list_codex_backends()["models"]
            claude = next(model for model in models if model["id"] == "claude-sonnet-5")
            self.assertEqual(claude["runner"], "claude")
            self.assertTrue(claude["available"])
            self.assertEqual(claude["serviceTiers"], ["default"])
            self.assertEqual(claude["label"], "Claude Sonnet 5")
            self.assertEqual(
                claude["efforts"], ["low", "medium", "high", "xhigh", "max"]
            )
            opus = next(model for model in models if model["id"] == "claude-opus-5")
            self.assertEqual(opus["label"], "Claude Opus 5")
            self.assertEqual(opus["efforts"], claude["efforts"])
            self.assertFalse(any(model["id"] in {"sonnet", "opus"} for model in models))
            self.assertEqual(
                config.normalize_role_runtime("opus", "max")["model"], "claude-opus-5"
            )

    def test_role_homes_preserve_sessions_and_do_not_import_codex_auth(self):
        with (
            TemporaryDirectory() as tmp,
            patch.object(config, "AGENT_WORKSPACES_ROOT", Path(tmp)),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=True),
        ):
            one = docker._prepare_role_codex_home("w", "c", "assistant", "claude")
            (one / "session.jsonl").write_text("saved")
            docker._prepare_role_codex_home("w", "c", "assistant", "claude")
            two = docker._prepare_role_codex_home("w", "c", "implementer", "claude")
            self.assertNotEqual(one, two)
            self.assertEqual((one / "session.jsonl").read_text(), "saved")
            self.assertTrue((one / "skills").is_symlink())
            self.assertFalse((one / "config.toml").exists())

    def test_persisted_session_is_reused_only_by_its_family(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main" / "logs").mkdir(parents=True)
            store = Store(WorkspaceRegistry(root / "state", main_dir=root / "main"))
            runtime = {"assistant": {"model": "sonnet"}}
            store.create(
                "w_main",
                "c",
                "workspace-write",
                agent_mode="single",
                role_runtimes=runtime,
            )
            store.set_codex_session(
                "w_main", "c", "assistant", "saved", family="claude"
            )
            self.assertEqual(
                store.sessions_for_prompt("w_main", "c", "new-contract", runtime),
                {"assistant": "saved"},
            )
            self.assertEqual(
                store.sessions_for_prompt(
                    "w_main",
                    "c",
                    "new-contract",
                    {"assistant": {"model": "gpt-5.6-sol"}},
                ),
                {},
            )


if __name__ == "__main__":
    unittest.main()
