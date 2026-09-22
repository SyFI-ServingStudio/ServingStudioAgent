"""Codex command construction without configuration or filesystem reads."""

import json
from dataclasses import dataclass
from pathlib import PurePosixPath

from ...runtime.command import ExecutionEnvironment
from ..base import AgentRequest


@dataclass(frozen=True)
class CodexCommand:
    environment: ExecutionEnvironment
    catalog_filename: str | None = None
    inherited_environment: tuple[str, ...] = ()
    permission_profile: str = ":danger-full-access"
    approval_policy: str = "never"
    approvals_reviewer: str | None = None

    def __post_init__(self):
        if self.catalog_filename not in {None, "models_catalog.json"}:
            raise ValueError("unsupported Codex catalog filename")
        if not self.permission_profile:
            raise ValueError("Codex permission profile is required")
        if self.approval_policy not in {"never", "on-request"}:
            # `untrusted` is unsupported and `on-failure` deprecated since 0.155.
            raise ValueError("unsupported Codex approval policy")
        if self.approvals_reviewer not in {None, "user", "auto_review"}:
            raise ValueError("unsupported Codex approvals reviewer")
        if self.approvals_reviewer is not None and self.approval_policy == "never":
            # `never` tells the model not to request escalation at all, so a
            # reviewer would never be consulted. Rejecting the pair keeps a dead
            # setting from reading like an active boundary.
            raise ValueError("Codex approvals reviewer requires an approval policy")

    def build_tracked(
        self, request: AgentRequest, *, home: str, pid_file: str
    ) -> list[str]:
        return request.execution.command(
            self.arguments(request, home=home),
            environment={"CODEX_HOME": home},
            inherited=self.inherited_environment,
            pid_file=pid_file,
            label="vibesim-codex",
        )

    def build(self, request: AgentRequest, *, home: str) -> list[str]:
        """The untracked form: transport prefix plus argv, no pid wrapper."""
        return [
            *request.execution.prefix(
                environment={"CODEX_HOME": home},
                inherited=self.inherited_environment,
            ),
            *self.arguments(request, home=home),
        ]

    def arguments(self, request: AgentRequest, *, home: str) -> list[str]:
        selection = request.selection
        command = ["codex", "exec"]
        if request.session_id is not None:
            command.append("resume")
        command.extend(["-m", selection.model.model_id])
        settings = {
            "model_reasoning_effort": selection.effort,
        }
        if self.catalog_filename is not None:
            settings["model_catalog_json"] = str(
                PurePosixPath(home) / self.catalog_filename
            )
        if len(selection.model.service_tiers) > 1:
            settings["service_tier"] = selection.service_tier
        settings.update(
            {
                "mcp_servers.analyzer.command": request.execution.mcp_python,
                "mcp_servers.analyzer.args": [request.execution.mcp_server],
                "mcp_servers.analyzer.env.ANALYZER_MCP_SOURCE": request.execution.analyzer_source,
                "mcp_servers.analyzer.env.ANALYZER_MCP_BASE_URL": request.execution.analyzer_base_url,
                "mcp_servers.analyzer.env.VIBESIM_MANAGED_RUN_CONTEXT": request.execution.managed_context,
            }
        )
        for key, value in settings.items():
            command.extend(["-c", f"{key}={json.dumps(value, ensure_ascii=False)}"])
        # Retired `--dangerously-bypass-approvals-and-sandbox` for a named
        # permission profile. `-s`/`sandbox_mode` must stay absent: whenever the
        # older sandbox settings appear, Codex silently ignores
        # `default_permissions`, and a sandbox that fails open without saying so
        # is the worst outcome available here.
        posture = {
            "default_permissions": self.permission_profile,
            "approval_policy": self.approval_policy,
        }
        if self.approvals_reviewer is not None:
            posture["approvals_reviewer"] = self.approvals_reviewer
        for key, value in posture.items():
            command.extend(["-c", f"{key}={json.dumps(value, ensure_ascii=False)}"])
        command.extend(
            [
                "--skip-git-repo-check",
                "--json",
            ]
        )
        if request.structured_output:
            command.extend(["--output-schema", str(request.output_schema)])
        if request.session_id is not None:
            command.extend([request.session_id, "-"])
        return command
