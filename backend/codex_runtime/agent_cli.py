"""Dispatch a role call without coupling the turn loop to an agent CLI."""

from collections.abc import AsyncIterator

from .claude_cli import run_claude
from .codex_cli import run_codex
from .config import codex_model
from .exec_types import CodexEvent


async def run_agent(
    container: str, prompt: str, **options
) -> AsyncIterator[CodexEvent]:
    runner = codex_model(options["model_id"]).family.runner
    execute = {"codex": run_codex, "claude": run_claude}[runner]
    async for event in execute(container, prompt, **options):
        yield event
