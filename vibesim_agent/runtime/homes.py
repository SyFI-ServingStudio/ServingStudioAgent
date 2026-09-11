"""Map the stored session compatibility scope to the same isolated mount path."""

from hashlib import sha256
from pathlib import Path, PurePosixPath

from ..domain.roles import Role
from .invocation import InvocationHome


def role_home(
    conversation_root: Path, container_root: str, role: Role, session_scope: str
) -> InvocationHome:
    if (
        not conversation_root.is_absolute()
        or not PurePosixPath(container_root).is_absolute()
    ):
        raise ValueError("runtime roots must be absolute")
    if not session_scope:
        raise ValueError("session scope must not be empty")
    # Scope is opaque and may contain URLs or separators. Its full digest avoids
    # treating provider configuration text as a filesystem path.
    scope_directory = sha256(session_scope.encode("utf-8")).hexdigest()
    relative = PurePosixPath(Role(role).value) / scope_directory
    role_directory = conversation_root / Role(role).value
    if role_directory.is_symlink() or (role_directory / scope_directory).is_symlink():
        raise ValueError("role runtime directories must not be symlinks")
    return InvocationHome(
        conversation_root / str(relative),
        str(PurePosixPath(container_root) / relative),
    )
