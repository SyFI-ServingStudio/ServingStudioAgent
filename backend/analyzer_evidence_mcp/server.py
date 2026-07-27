#!/usr/bin/env python3
"""A dependency-free stdio MCP bridge for the Analyzer read-only HTTP API."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from mcp.server.fastmcp import FastMCP

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 20.0
LOCAL_START_TIMEOUT_SECONDS = 15.0
TOOL_NAME = "read_analyzer_resource"

ENDPOINT_GUIDE = """Read one existing Analyzer protocol-v1 JSON resource.

Start with /api/v1/sweeps, then read /api/v1/sweeps/{sweep_id}/payload to
discover ordered axes, coordinate domains, metric keys, and opaque run_id
values. Run resources include descriptor, summary, topology, model, workload,
and subjects/{subject_id}/{report|payload}. Subject ids include concurrency,
request-state, slo-general, throughput, utilization, batch, kv-occupancy,
kernel-input-distribution, kernel-time-share, optimality, and
workload-conservation. Deeper resources are:
/runs/{run_id}/workers/{pool_tag}/{worker_id}/operations[?offset=&limit=],
/operations/seek?...,
/iterations/{iter_id}/optimality-kernel-ladder,
/iterations/{iter_id}/optimality-waterfall,
/operations/{iter_id}/{batch_id}/{operation_id}/cost-tree, and a cost-tree
leaf's /kernel-throughput-analysis.

Set source="host" for an experiment selected in the Analyzer UI. Set
source="workspace" for simulations created inside this agent workspace.
Only relative GET paths below /api/v1/ are accepted. Values are returned exactly
from Analyzer; this tool does not estimate, aggregate, or reinterpret metrics."""

mcp = FastMCP(
    "VibeSim Analyzer",
    instructions=(
        "Use the read_analyzer_resource tool to discover and inspect existing "
        "simulation results. Never guess resource identifiers or metric values."
    ),
)


class AnalyzerToolError(RuntimeError):
    """An actionable, user-safe Analyzer tool failure."""


_local_analyzer_process: subprocess.Popen[bytes] | None = None
_local_analyzer_base_url: str | None = None
_loopback_opener = build_opener(ProxyHandler({}))


def _validated_external_base_url() -> str:
    configured = os.environ.get("ANALYZER_MCP_BASE_URL", "").rstrip("/")
    if not configured:
        raise AnalyzerToolError(
            "ANALYZER_MCP_BASE_URL is required when ANALYZER_MCP_SOURCE=external"
        )
    parsed = urlsplit(configured)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AnalyzerToolError("ANALYZER_MCP_BASE_URL must be an HTTP(S) origin")
    if parsed.username or parsed.password or parsed.path not in {"", "/"}:
        raise AnalyzerToolError(
            "ANALYZER_MCP_BASE_URL must not contain credentials or a path"
        )
    return configured


def _find_analyzer_binary(repository_root: Path) -> Path:
    configured = os.environ.get("ANALYZER_MCP_ANALYZE_BIN", "").strip()
    candidates = (
        [Path(configured)]
        if configured
        else [
            repository_root / "target" / "release" / "analyze",
            repository_root / "target" / "debug" / "analyze",
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise AnalyzerToolError(
        "local Analyzer binary is unavailable; expected target/{release,debug}/analyze"
    )


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _stop_local_analyzer() -> None:
    global _local_analyzer_process
    if _local_analyzer_process is None or _local_analyzer_process.poll() is not None:
        return
    _local_analyzer_process.terminate()
    try:
        _local_analyzer_process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        _local_analyzer_process.kill()
        _local_analyzer_process.wait(timeout=3)


atexit.register(_stop_local_analyzer)


def _local_base_url() -> str:
    global _local_analyzer_base_url, _local_analyzer_process
    if (
        _local_analyzer_base_url is not None
        and _local_analyzer_process is not None
        and _local_analyzer_process.poll() is None
    ):
        return _local_analyzer_base_url

    repository_root = Path(
        os.environ.get("ANALYZER_MCP_REPO_ROOT", "/workspace")
    ).resolve()
    logs_root = Path(
        os.environ.get("ANALYZER_MCP_LOGS_ROOT", str(repository_root / "logs"))
    ).resolve()
    if not logs_root.is_dir():
        raise AnalyzerToolError(
            f"local simulation logs are unavailable at {logs_root}; run a simulation first"
        )
    analyzer_binary = _find_analyzer_binary(repository_root)
    loopback_port = _reserve_loopback_port()
    _local_analyzer_base_url = f"http://127.0.0.1:{loopback_port}"
    _local_analyzer_process = subprocess.Popen(
        [
            str(analyzer_binary),
            "serve",
            "--logs-root",
            str(logs_root),
            "--bind",
            f"127.0.0.1:{loopback_port}",
        ],
        cwd=repository_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + LOCAL_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _local_analyzer_process.poll() is not None:
            raise AnalyzerToolError(
                f"local Analyzer exited with code {_local_analyzer_process.returncode}"
            )
        try:
            # Readiness is a TCP concern. `/sweeps` performs real catalog work
            # and can legitimately take seconds on a large logs tree.
            with socket.create_connection(("127.0.0.1", loopback_port), timeout=0.5):
                return _local_analyzer_base_url
        except OSError:
            time.sleep(0.05)
    _stop_local_analyzer()
    raise AnalyzerToolError("local Analyzer did not become ready within 15 seconds")


def _base_url(source: str | None = None) -> str:
    selected_source = (
        (source or os.environ.get("ANALYZER_MCP_SOURCE", "external")).strip().lower()
    )
    if selected_source in {"workspace", "local"}:
        return _local_base_url()
    if selected_source in {"host", "external"}:
        return _validated_external_base_url()
    raise AnalyzerToolError(
        "source must be 'host' or 'workspace' "
        "(legacy 'external' and 'local' are also accepted)"
    )


def validate_resource_path(resource_path: str) -> str:
    """Keep the generic adapter inside the Analyzer's read-only protocol tree."""
    if not isinstance(resource_path, str) or not resource_path.startswith("/api/v1/"):
        raise AnalyzerToolError("path must start with /api/v1/")
    parsed = urlsplit(resource_path)
    if parsed.scheme or parsed.netloc or parsed.fragment:
        raise AnalyzerToolError("path must be relative and must not contain a fragment")
    decoded_segments = unquote(parsed.path).split("/")
    if any(segment in {".", ".."} for segment in decoded_segments):
        raise AnalyzerToolError("path traversal is not allowed")
    if len(resource_path) > 4096:
        raise AnalyzerToolError("path is too long")
    return resource_path


@mcp.tool(name=TOOL_NAME, description=ENDPOINT_GUIDE)
def read_analyzer_resource(
    path: str,
    source: Literal["host", "workspace"] = "host",
) -> Any:
    """Fetch and decode one bounded JSON response without changing its values."""
    safe_path = validate_resource_path(path)
    request_url = urljoin(f"{_base_url(source)}/", safe_path.lstrip("/"))
    request = Request(request_url, headers={"Accept": "application/json"}, method="GET")
    request_hostname = urlsplit(request_url).hostname
    open_request = (
        _loopback_opener.open
        if request_hostname in {"127.0.0.1", "::1", "localhost"}
        else urlopen
    )
    try:
        with open_request(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get_content_type()
            response_bytes = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        detail = error.read(2048).decode("utf-8", errors="replace")
        raise AnalyzerToolError(
            f"Analyzer returned HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise AnalyzerToolError(f"Analyzer request failed: {error.reason}") from error
    if len(response_bytes) > MAX_RESPONSE_BYTES:
        raise AnalyzerToolError("Analyzer response exceeded the 8 MiB tool limit")
    if content_type != "application/json":
        raise AnalyzerToolError(
            f"Analyzer returned unsupported content type {content_type!r}"
        )
    try:
        return json.loads(response_bytes)
    except json.JSONDecodeError as error:
        raise AnalyzerToolError("Analyzer returned invalid JSON") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--get",
        metavar="PATH",
        help="read one resource directly; useful for diagnostics and agent-tool tests",
    )
    arguments = parser.parse_args()
    if arguments.get:
        json.dump(
            read_analyzer_resource(arguments.get),
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        sys.stdout.write("\n")
        return
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
