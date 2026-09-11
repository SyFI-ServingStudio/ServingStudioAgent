"""Managed service startup using an explicit legacy deployment configuration."""

from contextlib import contextmanager
from pathlib import Path

from tools.migrate_v1_database import MigrationError, ProviderIdentity, migrate_database
from tools.migrate_v1_workspaces import (
    _database_scratch,
    _descriptors,
    _home_plan,
    _mapped_descriptors,
    _overlap,
    _sessions,
)
from tools.startup_migration import prepare_startup
from tools.startup_selection import (
    _location,
    _mapping,
    _outside_repositories,
    _providers,
    _read,
    _root,
    load_selection,
)
from tools.startup_state import inspect_state
from tools.tmux_deployment import TmuxDeployment

from .bootstrap import _source, configuration
from .providers.builtin import session_scope


class ManagedStartup:
    """Validate deployment inputs before any legacy shutdown or target writes."""

    def __init__(self, path, *, environment):
        self.environment = dict(environment)
        path = self._path(path)
        _, config = _read(path)
        if (
            set(config)
            != {
                "format",
                "source",
                "target",
                "selection",
                "receipt",
                "deployment",
                "mapping",
                "mode",
            }
            or type(config["format"]) is not int
            or config["format"] != 1
        ):
            raise MigrationError("unsupported managed startup configuration")
        self.source = Path(_root(self._path(config["source"]))["path"])
        self.target = self._path(config["target"])
        self.selection = self._path(config["selection"])
        self.receipt = self._path(config["receipt"])
        deployment = self._path(config["deployment"])
        mapping = self._path(config["mapping"])
        paths = [path, self.selection, self.receipt, deployment, mapping]
        if len(set(paths)) != len(paths):
            raise MigrationError("managed startup evidence paths must be distinct")
        if not isinstance(config["mode"], str) or config["mode"] not in {
            "production",
            "rehearsal",
        }:
            raise MigrationError("invalid managed startup migration mode")
        _, values = _read(mapping)
        if not {"models", "families", "runners"} <= set(values) or set(values) - {
            "models",
            "families",
            "runners",
            "external_paths",
        }:
            raise MigrationError("invalid managed startup provider mapping")
        try:
            identities = {
                kind: {
                    key: ProviderIdentity(**value)
                    for key, value in values[kind].items()
                }
                for kind in ("models", "families")
            }
        except (TypeError, AttributeError) as error:
            raise MigrationError(
                "invalid managed startup provider identities"
            ) from error
        self.options = {
            **identities,
            "runners": values["runners"],
            "mode": config["mode"],
        }
        _mapping(**identities)
        if not isinstance(values["runners"], dict) or any(
            not isinstance(value, str) for value in values["runners"].values()
        ):
            raise MigrationError("invalid managed startup runner mapping")
        _providers(identities["families"], values["runners"])
        self.external_paths = values.get("external_paths", {})
        if not isinstance(self.external_paths, dict) or any(
            not isinstance(paths, dict)
            or any(not isinstance(value, str) for value in paths.values())
            for paths in self.external_paths.values()
        ):
            raise MigrationError("invalid managed startup external paths")
        original = _descriptors(self.source)
        descriptors = _mapped_descriptors(
            self.source, original, config["mode"], self.external_paths
        )
        self.descriptors = descriptors
        for identity, descriptor in [*original.items(), *descriptors.items()]:
            if descriptor["storage_kind"] == "external":
                for key in ("repo_path", "logs_path"):
                    external = (self.source / identity / descriptor[key]).resolve()
                    if _overlap(self.target, external):
                        raise MigrationError(
                            "migration target overlaps external workspace"
                        )
        for evidence in paths:
            _location(evidence, self.source, self.target)
            _outside_repositories(evidence, self.source, {"descriptors": descriptors})
        _, self.report = _read(deployment)
        if self.report.get("source") != _root(self.source):
            raise MigrationError("managed deployment source differs from configuration")
        self._validate_runtime()
        if inspect_state(self.source) == "legacy-v8":
            for identity in descriptors:
                with _database_scratch(self.source / identity) as database:
                    migrate_database(database, None, **identities)
                    _home_plan(
                        self.source,
                        identity,
                        _sessions(database),
                        identities["families"],
                        values["runners"],
                    )

    def _validate_runtime(self):
        effective = self.environment | {
            "VIBESIM_AGENT_WORKSPACES_ROOT": str(self.target)
        }
        settings = configuration(environment=effective)
        _source(settings, effective)
        main = (
            self.source / "w_main" / self.descriptors["w_main"]["repo_path"]
        ).resolve()
        if main != settings.agent.main_dir.resolve():
            raise MigrationError("managed w_main differs from configured main checkout")
        runners = _providers(self.options["families"], self.options["runners"])
        for identity in [
            *self.options["models"].values(),
            *self.options["families"].values(),
        ]:
            if identity.provider_id not in settings.providers:
                raise MigrationError(
                    "migration provider is absent from runtime configuration"
                )
            runner = "claude" if identity.provider_id == "claude" else "codex"
            if runners.get(identity.provider_id, runner) != runner:
                raise MigrationError("migration runner differs from built-in runtime")
            if (
                session_scope(settings, identity.provider_id, runner)
                != identity.session_scope
            ):
                raise MigrationError(
                    "migration session scope differs from runtime configuration"
                )

    @staticmethod
    def _path(value):
        if not isinstance(value, (str, Path)):
            raise MigrationError("managed startup paths must be absolute")
        path = Path(value)
        if not path.is_absolute() or path.is_symlink():
            raise MigrationError(
                "managed startup paths must be absolute and non-symlink"
            )
        return path.parent.resolve(strict=True) / path.name

    def selected_root(self):
        """Read the published target for Analyzer; never initiate migration."""
        selected = load_selection(self.selection, self.source, **self.options)
        if selected is not None:
            if selected != self.target:
                raise MigrationError("configured target differs from startup selection")
            _, report = _read(selected / ".migration-v1/manifest.json")
            if report.get("descriptors") != self.descriptors:
                raise MigrationError("migration external paths changed")
            return selected
        if inspect_state(self.source) == "current":
            return self.source
        raise MigrationError("managed startup has not published a migrated target")

    @contextmanager
    def prepare(self):
        # Construction is lazy so current-format state needs no Docker access.
        @contextmanager
        def quiesce(source):
            owner = TmuxDeployment(
                self.report,
                environment=self.environment,
                receipt=self.receipt,
                target=self.target,
            )
            with owner.quiesce(source):
                yield

        with prepare_startup(
            self.source,
            self.target,
            self.selection,
            **self.options,
            external_paths=self.external_paths,
            quiesce=quiesce,
        ) as selected:
            # Copying a workspace can outlast a profile edit. Do not start an
            # application with scopes different from the ones just migrated.
            self._validate_runtime()
            yield self.environment | {"VIBESIM_AGENT_WORKSPACES_ROOT": str(selected)}
