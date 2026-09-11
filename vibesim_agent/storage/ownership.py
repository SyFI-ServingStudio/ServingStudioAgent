"""Process ownership of one stable workspace registry directory."""

import fcntl
import os
from pathlib import Path


class WorkspaceOwnership:
    def __init__(self, root: Path):
        if not root.is_absolute():
            raise ValueError("workspace ownership requires an absolute state root")
        self.root = root.resolve()
        self._descriptor: int | None = None

    def acquire(self) -> None:
        if self._descriptor is not None:
            return
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise RuntimeError("workspace state is owned by another backend") from error
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor

    def close(self) -> None:
        if self._descriptor is not None:
            descriptor, self._descriptor = self._descriptor, None
            os.close(descriptor)
