"""Explicit initialization and single-process service commands."""

import argparse
import json
import os
import sqlite3
from contextlib import nullcontext

import uvicorn

from tools.migrate_v1_database import MigrationError

from .bootstrap import create_application, host_home, initialize_state
from .providers.builtin import provider_environments
from .settings import ConfigurationError, environment_reference
from .startup import ManagedStartup
from .storage.database import SchemaMismatch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m vibesim_agent")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Initialize a new workspace state directory")
    serve = commands.add_parser("serve", help="Serve initialized or managed state")
    serve.add_argument("--startup-config", help="Absolute managed deployment JSON path")
    selected = commands.add_parser(
        "selected-root", help="Read managed state root for Analyzer"
    )
    selected.add_argument("--startup-config", required=True)
    commands.add_parser(
        "env-reference", help="Print supported environment configuration"
    )
    arguments = parser.parse_args(argv)
    environment = dict(os.environ)
    try:
        if arguments.command == "env-reference":
            print(environment_reference(provider_environments(host_home(environment))))
        elif arguments.command == "init":
            try:
                descriptor = initialize_state(environment=environment)
            except FileExistsError:
                raise ConfigurationError(
                    "VIBESIM_AGENT_WORKSPACES_ROOT already exists; init requires a new directory"
                ) from None
            print(json.dumps(descriptor))
        elif arguments.command == "selected-root":
            print(
                ManagedStartup(
                    arguments.startup_config, environment=environment
                ).selected_root()
            )
        else:
            startup = (
                ManagedStartup(
                    arguments.startup_config, environment=environment
                ).prepare()
                if arguments.startup_config
                else nullcontext(environment)
            )
            with startup as effective:
                app = create_application(environment=effective)
                try:
                    uvicorn.run(
                        app,
                        host=app.state.settings.agent.bind,
                        port=app.state.settings.agent.port,
                        workers=1,
                        lifespan="on",
                    )
                finally:
                    # Also release ownership if Uvicorn fails before lifespan.
                    app.state.recovery.close()
    except (ConfigurationError, SchemaMismatch, MigrationError) as error:
        parser.exit(2, f"{error}\n")
    except (OSError, sqlite3.Error) as error:
        parser.exit(2, f"startup failed ({type(error).__name__})\n")


if __name__ == "__main__":
    main()
