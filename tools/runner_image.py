"""Build a runner from the same explicit configuration used by the Agent."""

import argparse
import hashlib
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

from vibesim_agent.bootstrap import configuration
from vibesim_agent.settings import ConfigurationError

BUILD_OPTIONS = {
    "CUDA_IMAGE": "nvidia/cuda:12.8.1-devel-ubuntu24.04",
    "UV_IMAGE": "ghcr.io/astral-sh/uv:python3.12-bookworm",
    "CODEX_NPM_PACKAGE": "@openai/codex@0.155.1",
    "CLAUDE_NPM_PACKAGE": "@anthropic-ai/claude-code@2.1.278",
    "NODE_VERSION": "v22.23.2",
    "NODE_ARCH": "linux-x64",
    "RUST_TOOLCHAIN": "stable",
}


def build(
    *,
    environment: Mapping[str, str],
    repo_root: Path,
    run: Callable = subprocess.run,
) -> None:
    settings = configuration(environment=environment, repo_root=repo_root)
    container = settings.container
    fixed = {
        "VIBESIM_RUNNER_HOME": (container.home, Path("/home") / container.user),
        "VIBESIM_RUNNER_UV_PROJECT_ENVIRONMENT": (
            container.uv_project_environment,
            Path("/opt/vibesim-venv"),
        ),
        "VIBESIM_RUNNER_UV_CACHE_DIR": (
            container.uv_cache_dir,
            Path("/opt/vibesim-uv-cache"),
        ),
        "VIBESIM_RUNNER_DG_USE_LOCAL_VERSION": (container.dg_use_local_version, False),
    }
    for name, (actual, supported) in fixed.items():
        if actual != supported:
            raise ConfigurationError(
                f"Runner Dockerfile does not support configured {name}"
            )
    options = {}
    for name, default in BUILD_OPTIONS.items():
        key = "VIBESIM_RUNNER_" + name
        value = environment.get(key, default).strip()
        if not value:
            raise ConfigurationError(f"Invalid configuration: {key}")
        options[name] = value
    skip = environment.get("VIBESIM_RUNNER_SKIP_IMAGE_TEST", "0").strip()
    if skip not in {"0", "1"}:
        raise ConfigurationError("VIBESIM_RUNNER_SKIP_IMAGE_TEST must be 0 or 1")

    process_environment = {
        key: value for key, value in environment.items() if not key.startswith("GIT_")
    }
    process_environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    source = settings.agent.main_dir.resolve()
    top = run(
        ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
        env=process_environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if Path(top).resolve() != source:
        raise ConfigurationError("VIBESIM_AGENT_MAIN_DIR must be a Git repository root")
    lock = source / "uv.lock"
    if not lock.is_file():
        raise ConfigurationError("VIBESIM_AGENT_MAIN_DIR requires uv.lock")
    options.update(
        APP_UID=str(container.uid),
        APP_GID=str(container.gid),
        APP_USER=container.user,
        RUNNER_VERSION=container.version,
        VIBESIM_LOCK_SHA=hashlib.sha256(lock.read_bytes()).hexdigest(),
    )

    with TemporaryDirectory(
        prefix="vibesim-runner-build-", dir=environment.get("TMPDIR") or None
    ) as directory:
        context = Path(directory)
        run(
            [
                "bash",
                "-euo",
                "pipefail",
                "-c",
                'source "$1"; copy_main_tree "$2" "$3"',
                "runner-build",
                str(repo_root / "scripts/lib/main-tree-copy.sh"),
                str(source),
                str(context / "vibesim"),
            ],
            env=process_environment,
            check=True,
        )
        run(
            [
                "docker",
                "build",
                "-f",
                str(repo_root / "docker/runner.Dockerfile"),
                "-t",
                container.image,
                *(
                    arg
                    for key, value in options.items()
                    for arg in ("--build-arg", f"{key}={value}")
                ),
                str(context),
            ],
            env=process_environment,
            check=True,
        )
    if skip == "0":
        run(
            [
                "bash",
                str(repo_root / "scripts/test-runner-image.sh"),
                "build",
                "--main-dir",
                str(source),
            ],
            env=process_environment,
            check=True,
        )


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        build(environment=os.environ, repo_root=Path(__file__).resolve().parents[1])
    except (ConfigurationError, subprocess.CalledProcessError, OSError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
