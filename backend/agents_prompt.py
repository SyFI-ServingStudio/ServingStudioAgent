"""Render the generated prompt artifacts under `backend/prompts/` from templates.

Two runtime prompt families are generated, both keyed by the conversation's
`agent_mode` (and, for the workspace contract, by `autonomous`):

- the workspace contract bind-mounted read-only at `/workspace/AGENTS.md`
- the short driving-role prompt prefixed onto every Codex call

The rendered files are **build output, not source**: they are gitignored, and
`codex_runtime/config` calls `ensure_rendered()` when it is imported, so any
entry point (`run.sh`, a bare `uvicorn`, `unittest discover`) materializes them
before the first reader. They must exist as real files because the runtime
consumes them as such — `codex_runtime/docker.py` bind-mounts the contract into
the container, `codex_runtime/prompts.py` reads the role prompt, and
`codex_runtime/config.prompt_fingerprint` hashes their bytes.

Agent-facing rule: never hand-edit a file under `backend/prompts/` that appears
in one of the matrices below — the next import overwrites it. Edit
`backend/prompt_templates/`, then read the result:

    uv run python -m backend.agents_prompt

The remaining files in `backend/prompts/` (the JSON schemas, `implementer.txt`,
`naming-*.txt`) have no mode variants, are hand-written, and stay tracked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATES_DIR = Path(__file__).resolve().parent / "prompt_templates"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

AGENTS_TEMPLATE = "AGENTS.md.j2"
ROLE_TEMPLATE = "role.txt.j2"

# (agent_mode, autonomous) -> rendered workspace-contract filename. The two
# orchestrated names predate templating and must keep their spelling: they are
# what `codex_runtime/config.agents_prompt_name` returns and what the container
# reuse check compares against.
AGENTS_PROMPT_MATRIX: dict[tuple[str, bool], str] = {
    ("orchestrated", False): "AGENTS.md",
    ("orchestrated", True): "AGENTS.autonomous.md",
    ("single", False): "AGENTS.single.md",
    ("single", True): "AGENTS.single.autonomous.md",
}

# agent_mode -> rendered driving-role prompt filename. `implementer.txt` is not
# generated: it has no mode variants.
ROLE_PROMPT_MATRIX: dict[str, str] = {
    "orchestrated": "orchestrator.txt",
    "single": "assistant.txt",
}


def _environment() -> Environment:
    # trim_blocks/lstrip_blocks let a `{% if %}` sit on its own line without
    # leaving a blank line behind, which is what makes the orchestrated
    # artifacts byte-identical to their hand-written predecessors.
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_agents_prompt(*, agent_mode: str, autonomous: bool) -> str:
    """Render the `/workspace/AGENTS.md` contract for one mode combination."""
    template = _environment().get_template(AGENTS_TEMPLATE)
    return template.render(agent_mode=agent_mode, autonomous=autonomous)


def render_role_prompt(*, agent_mode: str) -> str:
    """Render the per-call driving-role prompt for one agent mode."""
    template = _environment().get_template(ROLE_TEMPLATE)
    return template.render(agent_mode=agent_mode)


def rendered_artifacts() -> dict[str, str]:
    """Map every generated filename to its freshly rendered text."""
    artifacts = {
        filename: render_agents_prompt(agent_mode=agent_mode, autonomous=autonomous)
        for (agent_mode, autonomous), filename in AGENTS_PROMPT_MATRIX.items()
    }
    artifacts.update(
        {
            filename: render_role_prompt(agent_mode=agent_mode)
            for agent_mode, filename in ROLE_PROMPT_MATRIX.items()
        }
    )
    return artifacts


def ensure_rendered() -> list[str]:
    """Materialize every generated file; return the names whose bytes changed.

    Idempotent and cheap: a fresh tree re-renders six small templates and writes
    nothing. Called at `codex_runtime/config` import so no reader can observe a
    missing or stale artifact, which is what lets these files be gitignored.
    """
    changed = []
    for filename, text in sorted(rendered_artifacts().items()):
        path = PROMPTS_DIR / filename
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        path.write_text(text, encoding="utf-8")
        changed.append(filename)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    changed = ensure_rendered()
    for filename in changed:
        print(f"wrote: backend/prompts/{filename}")
    if not changed:
        print(f"unchanged: {len(rendered_artifacts())} generated prompt files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
