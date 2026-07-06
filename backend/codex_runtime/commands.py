"""Small subprocess helpers shared by workspace and Docker setup."""

from __future__ import annotations

import subprocess

def run_checked(cmd: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"{(result.stderr or result.stdout).strip()[-2000:]}"
        )
    return result
