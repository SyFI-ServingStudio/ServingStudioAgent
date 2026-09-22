"""Create a ServingStudioSim worktree from the Agent's own configuration."""

import argparse
import os
from pathlib import Path

from vibesim_agent.bootstrap import configuration
from vibesim_agent.runtime.worktree import WorktreeError, WorktreeProvisioner
from vibesim_agent.settings import ConfigurationError


def create(
    *,
    environment,
    repo_root: Path,
    topic: str,
    branch: str | None = None,
    base: str | None = None,
    root: Path | None = None,
):
    settings = configuration(environment=environment, repo_root=repo_root)
    main = settings.agent.main_dir.resolve()
    # The convention is a sibling of the main checkout, never nested inside it.
    destination = (root or main.parent).resolve() / f"wt-{topic}"
    provisioner = WorktreeProvisioner(main, process_environment=environment)
    return provisioner.create(destination, branch=branch or f"wt-{topic}", base=base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", help="short kebab-case topic; the tree is wt-<topic>")
    parser.add_argument("--branch", help="branch name (default: wt-<topic>)")
    parser.add_argument("--base", help="base ref (default: the main checkout's HEAD)")
    parser.add_argument(
        "--root", type=Path, help="directory to hold the worktree (default: sibling)"
    )
    arguments = parser.parse_args()
    try:
        worktree = create(
            environment=os.environ,
            repo_root=Path(__file__).resolve().parents[1],
            topic=arguments.topic,
            branch=arguments.branch,
            base=arguments.base,
            root=arguments.root,
        )
    except (ConfigurationError, WorktreeError, ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
    print(f"{worktree.path}\t{worktree.branch}\t{worktree.base_revision}")
    print(
        "First `uv run` / `just test-*` inside it will sync and build once; "
        "run every command from inside the worktree."
    )


if __name__ == "__main__":
    main()
