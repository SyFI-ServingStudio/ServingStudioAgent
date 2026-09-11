"""Coordinate migration while the deployment owner suppresses legacy writers."""

from contextlib import contextmanager
from pathlib import Path

from tools.migrate_v1_database import MigrationError
from tools.migrate_v1_workspaces import (
    _descriptors,
    _mapped_descriptors,
    migrate_workspaces,
)
from tools.startup_selection import (
    _location,
    _mapping,
    _outside_repositories,
    _providers,
    _read,
    _root,
    load_selection,
    publish_selection,
)
from tools.startup_state import inspect_state
from vibesim_agent.storage.ownership import WorkspaceOwnership


@contextmanager
def prepare_startup(
    source,
    target,
    selection,
    *,
    models,
    families,
    runners,
    quiesce=None,
    mode="production",
    external_paths=None,
):
    """Yield the state root to serve, retaining legacy suppression until exit.

    ``quiesce(source)`` is a deployment-owned context manager that stops every
    legacy writer and prevents its restart. Its exit MUST NOT restart the old
    service: even an exception in the yielded body does not authorize rollback.
    This helper cannot verify an arbitrary callback's stop implementation.

    The application acquires its own target lock. Our source lock only excludes
    other new coordinators; the legacy backend does not participate in it.
    """
    source = Path(_root(source)["path"])
    target = Path(target)
    if not target.is_absolute() or target.is_symlink():
        raise MigrationError("migration target requires an absolute non-symlink path")
    target = target.parent.resolve(strict=True) / target.name
    selection = _location(selection, source, target)
    options = {"models": models, "families": families, "runners": runners, "mode": mode}
    _mapping(models, families)
    _providers(families, runners)
    if mode not in {"production", "rehearsal"}:
        raise MigrationError("migration mode must be production or rehearsal")

    # Never let format detection bypass an existing, possibly invalid record.
    if (
        not selection.exists()
        and not selection.is_symlink()
        and inspect_state(source) == "current"
    ):
        yield source
        return
    if quiesce is None:
        raise MigrationError("legacy startup requires deployment-owned writer shutdown")

    ownership = WorkspaceOwnership(source)
    ownership.acquire()
    try:
        selected = load_selection(selection, source, **options)
        if selected is not None and selected != target:
            raise MigrationError("configured target differs from startup selection")
        with quiesce(source):
            if inspect_state(source) != "legacy-v8":
                raise MigrationError("migration source is no longer legacy state")
            descriptors = _mapped_descriptors(
                source, _descriptors(source), mode, external_paths or {}
            )
            _outside_repositories(selection, source, {"descriptors": descriptors})
            selected = load_selection(selection, source, **options)
            if selected is not None and selected != target:
                raise MigrationError("configured target differs from startup selection")
            if selected is None and not target.exists():
                migrate_workspaces(
                    source,
                    target,
                    **options,
                    external_paths=external_paths,
                    dry_run=False,
                    source_quiesced=True,
                )
            # A crash before publication may leave a complete target. Only
            # the same full verification may adopt it; partial copies fail.
            _, report = _read(target / ".migration-v1/manifest.json")
            if report.get("descriptors") != descriptors:
                raise MigrationError("migration external paths changed")
            if selected is None:
                selected = publish_selection(selection, source, target, **options)
            yield selected
    finally:
        ownership.close()
