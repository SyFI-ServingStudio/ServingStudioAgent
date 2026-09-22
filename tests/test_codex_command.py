import json
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path

from tests.runtime_fixtures import (
    agent_request,
    docker_execution,
    execution_environment,
)
from vibesim_agent.domain.roles import Role
from vibesim_agent.providers.base import OutputMode
from vibesim_agent.providers.codex.command import CodexCommand


def command_golden():
    return json.loads(
        (Path(__file__).parent / "fixtures/legacy_commands/golden.json").read_text()
    )


def normalize_legacy_gpu_environment(command):
    return [
        "VIBESIM_RUNNER_GPUS=" + arg.removeprefix("CODEX_DOCKER_GPUS=")
        if index > 0
        and command[index - 1] == "-e"
        and arg.startswith("CODEX_DOCKER_GPUS=")
        else arg
        for index, arg in enumerate(command)
    ]


class CodexCommandTests(unittest.TestCase):
    def test_inherited_credentials_do_not_override_runtime_values(self):
        runtime = self.builder().environment
        with self.assertRaisesRegex(ValueError, "conflicts"):
            runtime.prefix("container", environment={}, inherited=("USER",))
        command = runtime.prefix(
            "container", environment={}, inherited=("PROVIDER_API_KEY",)
        )
        self.assertIn("PROVIDER_API_KEY", command)
        self.assertFalse(
            any(value.startswith("PROVIDER_API_KEY=") for value in command)
        )

    def builder(self):
        return CodexCommand(
            replace(
                execution_environment(),
                managed_context="/home/runner/.vibesim-codex/managed-run.json",
            )
        )

    def request(self):
        return replace(
            agent_request(),
            execution_id="fixed-execution",
            execution=docker_execution(self.builder().environment),
        )

    def test_fresh_and_resume_match_old_command(self):
        cases = command_golden()["cases"]
        self.assertEqual(
            {(case["role"], case["session_id"]) for case in cases},
            {
                (role.value, session)
                for role in Role
                for session in (None, "saved-session")
            },
        )
        for case in cases:
            with self.subTest(role=case["role"], session=case["session_id"]):
                request = replace(
                    self.request(),
                    role=Role(case["role"]),
                    session_id=case["session_id"],
                )
                self.assertEqual(
                    self.builder().build(request, home=case["home"]),
                    normalize_legacy_gpu_environment(case["codex"]),
                )

    def test_prompt_model_omits_schema_and_unsupported_tier(self):
        request = self.request()
        request = replace(
            request,
            selection=replace(
                request.selection,
                model=replace(
                    request.selection.model,
                    output_mode=OutputMode.PROMPT,
                    service_tiers=("default",),
                ),
                service_tier="default",
            ),
        )
        command = self.builder().build(request, home="/role-home")
        self.assertNotIn("--output-schema", command)
        self.assertFalse(any(arg.startswith("service_tier=") for arg in command))

    def test_config_paths_are_toml_quoted_and_prompt_stays_out_of_argv(self):
        execution = replace(
            docker_execution(self.builder().environment),
            mcp_python='/quoted/"python',
            mcp_server="/space path/server.py",
        )
        request = replace(self.request(), execution=execution)
        command = self.builder().build(request, home="/role-home")
        values = [
            command[index + 1] for index, part in enumerate(command) if part == "-c"
        ]
        parsed = tomllib.loads("\n".join(values))
        self.assertEqual(
            parsed["mcp_servers"]["analyzer"]["command"], execution.mcp_python
        )
        self.assertNotIn(request.prompt, command)


class CodexSandboxPostureTests(unittest.TestCase):
    """Pins the posture that replaced `--dangerously-bypass-approvals-and-sandbox`."""

    def builder(self, **overrides):
        return replace(CodexCommandTests().builder(), **overrides)

    def settings(self, command):
        return tomllib.loads(
            "\n".join(
                command[index + 1]
                for index, part in enumerate(command)
                if part == "-c"
            )
        )

    def test_container_posture_is_explicit_and_retires_the_bypass_flag(self):
        command = self.builder().build(
            CodexCommandTests().request(), home="/role-home"
        )
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)
        settings = self.settings(command)
        self.assertEqual(settings["default_permissions"], ":danger-full-access")
        self.assertEqual(settings["approval_policy"], "never")
        self.assertNotIn("approvals_reviewer", settings)

    def test_legacy_sandbox_generation_is_never_emitted(self):
        # Codex ignores `default_permissions` outright whenever the older sandbox
        # settings are present, so their absence is the whole posture.
        command = self.builder(
            permission_profile="vibesim_host",
            approval_policy="on-request",
            approvals_reviewer="auto_review",
        ).build(CodexCommandTests().request(), home="/role-home")
        self.assertNotIn("-s", command)
        self.assertNotIn("--sandbox", command)
        self.assertNotIn("sandbox_mode", self.settings(command))
        self.assertFalse(
            any(arg.startswith("sandbox_workspace_write") for arg in command)
        )

    def test_host_posture_carries_the_reviewer(self):
        command = self.builder(
            permission_profile="vibesim_host",
            approval_policy="on-request",
            approvals_reviewer="auto_review",
        ).build(CodexCommandTests().request(), home="/role-home")
        settings = self.settings(command)
        self.assertEqual(settings["default_permissions"], "vibesim_host")
        self.assertEqual(settings["approval_policy"], "on-request")
        self.assertEqual(settings["approvals_reviewer"], "auto_review")

    def test_retired_and_dead_posture_values_are_rejected(self):
        for policy in ("untrusted", "on-failure", ""):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, "approval policy"):
                    self.builder(approval_policy=policy)
        with self.assertRaisesRegex(ValueError, "approvals reviewer"):
            self.builder(approvals_reviewer="everyone")
        with self.assertRaisesRegex(ValueError, "requires an approval policy"):
            # A reviewer under `never` is dead config that reads like a boundary.
            self.builder(approvals_reviewer="auto_review")
        with self.assertRaisesRegex(ValueError, "permission profile"):
            self.builder(permission_profile="")
