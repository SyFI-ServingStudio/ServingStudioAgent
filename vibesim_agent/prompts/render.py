"""Render role contracts explicitly into an application-owned output directory."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..domain.conversations import RoleRuntime
from ..domain.roles import AgentMode, Role


def agents_name(mode: AgentMode, autonomous: bool) -> str:
    mode_suffix = ".single" if mode is AgentMode.SINGLE else ""
    autonomous_suffix = ".autonomous" if autonomous else ""
    return f"AGENTS{mode_suffix}{autonomous_suffix}.md"


class Prompts:
    def __init__(self, directory: Path):
        self.directory = directory

    @classmethod
    def prepare(cls, directory: Path) -> Prompts:
        sources = Path(__file__).parent
        environment = Environment(
            loader=FileSystemLoader(sources / "templates"),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        artifacts = {}
        for mode in AgentMode:
            for autonomous in (False, True):
                artifacts[agents_name(mode, autonomous)] = (
                    environment.get_template("AGENTS.md.j2")
                    .render(agent_mode=mode.value, autonomous=autonomous)
                    .encode()
                )
            artifacts[f"{mode.driver.value}.txt"] = (
                environment.get_template("role.txt.j2")
                .render(agent_mode=mode.value)
                .encode()
            )
        for path in (sources / "contracts").iterdir():
            if path.is_file():
                artifacts[path.name] = path.read_bytes()
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in artifacts.items():
            path = directory / name
            if not path.is_file() or path.read_bytes() != content:
                path.write_bytes(content)
        return cls(directory)

    def role_text(self, role: Role) -> str:
        return (
            (self.directory / f"{role.value}.txt").read_text(encoding="utf-8").strip()
        )

    def schema_path(self, role: Role) -> Path:
        return self.directory / f"{role.value}.schema.json"

    def contract_path(self, mode: AgentMode, autonomous: bool) -> Path:
        return self.directory / agents_name(mode, autonomous)

    def driver_prompt(
        self, prompt: str, *, mode: AgentMode, conversation_id: str
    ) -> str:
        return (
            f"{self.driver_contract(mode, conversation_id)}"
            f"\n\nNewest user message:\n{prompt}\n"
        )

    def driver_contract(self, mode: AgentMode, conversation_id: str) -> str:
        return (
            f"{self.role_text(mode.driver)}\n\n"
            f"Current conversation ID: `{conversation_id}`.\n"
            f"Conversation plan: `/workspace/{conversation_id}_plan.md`.\n"
            f"Conversation progress: `/workspace/{conversation_id}_progress.md`."
        )

    def orchestrator_handoff_prompt(
        self,
        task: str,
        implementer_text: str,
        *,
        conversation_id: str,
    ) -> str:
        return (
            f"{self.driver_contract(AgentMode.ORCHESTRATED, conversation_id)}\n\n"
            "The implementer returned a summary for your delegated task.\n\n"
            "You do not share the implementer Codex session. Treat the text below as "
            "the explicit handoff record, review it against your own orchestration "
            "context, and return exactly one JSON object.\n\n"
            "If the work is complete, use `final_answer`. If clarification, "
            "authorization, or an external choice is genuinely required, use "
            "`request_user_input`. If another bounded code-change, validation, or "
            "large exploration task is still needed, use `delegate` with that "
            "specific follow-up task.\n\n"
            "Delegated task:\n"
            f"{task}\n\n"
            "Implementer summary:\n"
            f"{implementer_text}\n"
        )

    def driver_repair_prompt(
        self,
        unparsed_output: str,
        *,
        agent_mode: str,
        conversation_id: str,
    ) -> str:
        """Ask the resumed driving role to repair only its decision envelope."""
        if agent_mode == "single":
            envelope = (
                "Return exactly one JSON object with the fields `action` and "
                "`message`. If the text below is the completed answer, preserve it "
                "in `message` with action `final_answer`. If user input is genuinely "
                "required, use `request_user_input`. There is no `delegate` action "
                "and no `task` field in this mode."
            )
        else:
            envelope = (
                "Return exactly one JSON object with the fields `action`, `message`, "
                "and `task`. If the text below is the completed answer, preserve it "
                "in `message` with action `final_answer`. If user input is genuinely "
                "required, use `request_user_input`; if a separate implementer task "
                "is required, use `delegate`."
            )
        return (
            f"{self.driver_contract(AgentMode(agent_mode), conversation_id)}\n\n"
            "Your previous final output could not be parsed as the required decision "
            f"JSON. Do not redo completed analysis. {envelope} Do not end with "
            "a progress update and do not add text outside the JSON object.\n\n"
            f"Unparsed previous output:\n{unparsed_output}\n"
        )

    def driver_continue_prompt(
        self,
        action: str,
        message: str,
        *,
        agent_mode: str,
        conversation_id: str,
    ) -> str:
        """Resume after a non-terminal envelope was emitted as the final item."""
        terminal_actions = (
            "`final_answer` or `request_user_input`"
            if agent_mode == "single"
            else "`final_answer`, `request_user_input`, or `delegate`"
        )
        return (
            f"{self.driver_contract(AgentMode(agent_mode), conversation_id)}\n\n"
            f"Your previous call ended with a non-terminal `{action}` update. The "
            "runtime already showed it to the user. Continue the same work from that "
            "checkpoint without repeating completed analysis. End this call only with "
            f"{terminal_actions}; use `progress` and "
            "`milestone` only for commentary emitted while you keep working.\n\n"
            f"Last update:\n{message}\n"
        )

    def implementer_steer_prompt(self, message: str) -> str:
        """Deliver a user correction to an implementer they interrupted mid-task.

        Unlike `_implementer_prompt` this is not a new delegated task: the session
        being resumed still holds the original one, and the user stopped it
        precisely because they wanted that task done differently. Saying so is what
        keeps the model from re-reading and re-running work it already finished.

        The `user:` marker is load-bearing, not decoration: it is the only thing
        that distinguishes a message the user typed from a task the orchestrator
        delegated, and the implementer's contract keys `reply_user` off exactly
        that. Nothing else in an implementer prompt carries it.
        """
        return (
            f"{self.role_text(Role.IMPLEMENTER)}\n\n"
            "The user interrupted you while you were working on the task above, and "
            "sent the message below. Read it first: if it asks you something, answer "
            "it with `reply_user`. If it corrects how you were going about the task, "
            "continue from where you stopped — keep the work you already completed, "
            "drop or redo only what the message contradicts, and end with "
            "`final_answer` as usual.\n\n"
            f"user: {message}\n"
        )

    def implementer_prompt(
        self,
        task: str,
        *,
        is_resume: bool,
    ) -> str:
        # Keep the role contract explicit on every Codex call. A resumed session
        # has prior context, but the new delegated task must still be framed as
        # implementor work rather than relying on that context implicitly.
        del is_resume
        return f"{self.role_text(Role.IMPLEMENTER)}\n\nTask:\n{task}\n"

    def fingerprint(
        self, *, mode: AgentMode, autonomous: bool, runtimes: Mapping[Role, RoleRuntime]
    ) -> str:
        names = [
            agents_name(mode, autonomous),
            f"{mode.driver.value}.txt",
            f"{mode.driver.value}.schema.json",
        ]
        if Role.IMPLEMENTER in mode.roles:
            names.insert(2, "implementer.txt")
        digest = hashlib.sha256()
        for name in names:
            digest.update(
                name.encode() + b"\0" + (self.directory / name).read_bytes() + b"\0"
            )
        for role in mode.roles:
            runtime = runtimes[role]
            values = (
                role.value,
                runtime.provider_id,
                runtime.model_id,
                runtime.effort,
                runtime.service_tier,
            )
            digest.update(b"\0".join(value.encode() for value in values))
        digest.update(b"\0autonomous=" + str(autonomous).encode())
        digest.update(b"\0agent_mode=" + mode.value.encode())
        return digest.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    Prompts.prepare(args.output)


if __name__ == "__main__":
    main()
