"""The legacy composition root for transport-level acceptance tests.

Only this fixture knows where the old application keeps its dependencies.
The scenarios exercise HTTP and a scripted CLI, including the real turn loop.
"""

from __future__ import annotations

import asyncio
import importlib
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from unittest.mock import PropertyMock, patch

from backend.codex_runtime import config, turn
from backend.store import Store, WorkspaceRegistry


@asynccontextmanager
async def legacy_application(root: Path, runner):
    repo = root / "repo"
    (repo / "logs").mkdir(parents=True)
    registry = WorkspaceRegistry(root / "state", main_dir=repo)
    store = Store(registry)
    # Importing app constructs a Store. Intercept that construction before it
    # can initialize or migrate the host's real agent-workspaces directory.
    with patch("backend.store.Store", return_value=store):
        module = importlib.import_module("backend.app")
    evaluation = importlib.import_module("backend.eval")
    with ExitStack() as stack:
        for owner, name, value in (
            (module, "store", store),
            (module, "_workspace_locks", {}),
            (module, "_active_browser_turns", {}),
            (module, "VIBESIM_API_TOKEN", "contract-token"),
            (turn, "run_agent", runner),
        ):
            stack.enter_context(patch.object(owner, name, value))
        for owner, name, value in (
            (turn, "workspace_main_for", repo),
            (turn, "prepare_workspace", repo),
            (turn, "container_running", True),
            (turn, "ensure_container", "contract-container"),
            (turn, "write_managed_context", None),
            (module, "schedule_auto_naming", False),
            (module, "prepare_workspace", repo),
            (module, "cleanup_conversation", None),
            (module, "remove_managed_context", None),
            (evaluation, "WorkspaceRegistry", registry),
            (evaluation, "workspace_main_for", repo),
            (evaluation, "remove_container", None),
        ):
            stack.enter_context(patch.object(owner, name, return_value=value))
        stack.enter_context(
            patch.object(
                config.CodexFamilySpec,
                "available",
                new_callable=PropertyMock,
                return_value=True,
            )
        )
        try:
            yield module.app
        finally:
            # Drain before removing patches or deleting the fixture database,
            # even when a streaming assertion fails before it can send Stop.
            await asyncio.sleep(0)
            tasks = [
                active.task
                for active in module._active_browser_turns.values()
                if active.task is not None and not active.task.done()
            ]
            for task in tasks:
                if not task.cancelling():
                    task.cancel()
            if tasks:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=5
                )
