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
from urllib.parse import parse_qs, unquote, urljoin, urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from mcp.server.fastmcp import FastMCP

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_EVIDENCE_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 20.0
LOCAL_START_TIMEOUT_SECONDS = 15.0
TOOL_NAME = "read_analyzer_resource"

ENDPOINT_GUIDE = """Read one existing Analyzer protocol-v1 JSON resource.

Start with /api/v1/sweeps?status=ready&limit=5 for recent candidates, or
/api/v1/sweeps/latest for the newest ready candidate. Verify display_name,
ordered axes, deployment, trace, status, and time against the user's request;
latest does not by itself prove semantic relevance. Then read
/api/v1/sweeps/{sweep_id}/payload to discover coordinate domains, metric keys,
and opaque run_id values. Run resources include descriptor, summary, topology, model, workload,
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

Offline timing predictions are separate first-class resources. Start with
/predictions, then follow /predictions/{prediction_id}/descriptor and /cases.
Exact evidence is available below
/predictions/{prediction_id}/cases/{case_id}/operations/{operation_id}/cost-tree,
with the same leaf /kernel-throughput-analysis, case optimality-kernel-ladder,
case optimality-waterfall, and prediction-level kernel-input-distribution
resources. Prediction responses intentionally contain no pool or worker identity.

Kernel profiling and measurement results are first-class resources too. Start
with /kernel-profiles or /kernel-measurements, then follow their descriptor
links. A profile exposes /kernel-profiles/{profile_id}/curve. A measurement
exposes /kernel-measurements/{measurement_id}/summary and only the plot links
declared by its descriptor. Resolve GPU ceilings with
/hardware/gpus?name={gpu_name}; catalog TFLOPS are dense peaks and interconnect
bandwidth includes both bidirectional and derived one-way values.

Set source="host" for an experiment selected in the Analyzer UI. Set
source="workspace" for simulations created inside this agent workspace.
Only relative GET paths below /api/v1/ are accepted. Values are returned exactly
from Analyzer; this tool does not estimate, aggregate, or reinterpret metrics.

In a managed UI turn, reading an exact sweep payload from either source returns
one compact block: resource, ordered axes, metric metadata, and rows. Every raw
value has an adjacent complete citation token. Reading an exact timing-prediction
resource and exact run endpoint return one {citation, result} block. Kernel profile
and measurement resources return {result, citations}, keyed by the metric or plot
shown in the result. Copy the complete citation token
unchanged as Markdown inline code beside the supported claim. Do not assemble,
alter, or invent tokens. They become clickable only when the user clicks the
final answer."""

mcp = FastMCP(
    "VibeSim Analyzer",
    instructions=(
        "Use read_analyzer_resource to discover and inspect Analyzer-owned "
        "simulation sweeps, runs, timing predictions, kernel profiles, and "
        "kernel measurements. Exact typed reads return complete citation tokens; "
        "copy the matching token unchanged beside each supported claim. Never "
        "guess resource identifiers, metric values, or citation tokens."
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


def _managed_context() -> dict[str, Any] | None:
    context_value = os.environ.get("VIBESIM_MANAGED_RUN_CONTEXT", "").strip()
    if not context_value:
        return None
    context_path = Path(context_value)
    try:
        context = json.loads(context_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalyzerToolError("managed Agent context could not be read") from error
    if not isinstance(context, dict):
        raise AnalyzerToolError("managed Agent context is invalid")
    return context


def _exact_sweep_id(safe_path: str, payload: Any) -> str | None:
    """Return the path-bound sweep id only for an exact protocol payload read."""
    parsed_path = urlsplit(safe_path)
    path_segments = parsed_path.path.strip("/").split("/")
    if (
        len(path_segments) != 5
        or path_segments[:3] != ["api", "v1", "sweeps"]
        or path_segments[4] != "payload"
        or not isinstance(payload, dict)
    ):
        return None
    path_sweep_id = path_segments[3]
    payload_sweep_id = payload.get("sweep_id")
    if payload_sweep_id != path_sweep_id:
        raise AnalyzerToolError(
            "Analyzer sweep payload identity does not match its path"
        )
    return path_sweep_id


def _register_sweep_citations(
    safe_path: str,
    payload: Any,
) -> dict[str, Any] | None:
    """Register one exact sweep and return its host-issued dictionary."""
    experiment_id = _exact_sweep_id(safe_path, payload)
    if experiment_id is None:
        return None
    context = _managed_context()
    if context is None:
        return None
    backend_url = context.get("backend_url")
    capability_token = context.get("capability_token")
    if not isinstance(backend_url, str) or not isinstance(capability_token, str):
        raise AnalyzerToolError("managed Agent context is incomplete")
    request_body = json.dumps(
        {"experimentId": experiment_id, "analysis": payload},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    registration_url = urljoin(
        f"{backend_url.rstrip('/')}/",
        "api/internal/analyzer-citations/register",
    )
    registration_request = Request(
        registration_url,
        data=request_body,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {capability_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(
            registration_request,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            dictionary = json.loads(response.read(MAX_RESPONSE_BYTES + 1))
    except HTTPError as error:
        detail = error.read(2048).decode("utf-8", errors="replace")
        raise AnalyzerToolError(
            f"citation registration returned HTTP {error.code}: {detail}"
        ) from error
    except (URLError, json.JSONDecodeError) as error:
        raise AnalyzerToolError(f"citation registration failed: {error}") from error
    if not isinstance(dictionary, dict):
        raise AnalyzerToolError("citation registration returned an invalid dictionary")
    return dictionary


def _prediction_target_from_path(safe_path: str) -> dict[str, Any] | None:
    parsed_path = urlsplit(safe_path)
    segments = parsed_path.path.strip("/").split("/")
    if len(segments) < 5 or segments[:3] != ["api", "v1", "predictions"]:
        return None
    prediction_id = segments[3]
    suffix = segments[4:]
    target: dict[str, Any] = {
        "predictionId": prediction_id,
        "caseId": None,
        "operationId": None,
        "leafId": None,
        "panelId": None,
        "optimalityMode": "unlocked",
    }
    query = parse_qs(parsed_path.query)
    requested_mode = query.get("mode", ["unlocked"])[0]
    if requested_mode in {"unlocked", "batch_locked"}:
        target["optimalityMode"] = requested_mode
    if suffix in (["descriptor"], ["cases"]):
        return target
    if len(suffix) == 3 and suffix[0] == "cases" and suffix[2] in {
        "optimality-waterfall",
        "optimality-kernel-ladder",
    }:
        target["caseId"] = suffix[1]
        target["panelId"] = (
            "optimality-breakdown"
            if suffix[2] == "optimality-waterfall"
            else "optimality-kernel-ladder"
        )
        return target
    if (
        len(suffix) == 5
        and suffix[0] == "cases"
        and suffix[2] == "operations"
        and suffix[4] == "cost-tree"
    ):
        target.update(
            {"caseId": suffix[1], "operationId": suffix[3], "panelId": "cost-tree"}
        )
        return target
    if (
        len(suffix) == 7
        and suffix[0] == "cases"
        and suffix[2] == "operations"
        and suffix[4] == "cost-tree"
        and suffix[5].isdigit()
        and suffix[6] == "kernel-throughput-analysis"
    ):
        target.update(
            {
                "caseId": suffix[1],
                "operationId": suffix[3],
                "leafId": int(suffix[5]),
                "panelId": "kernel-throughput",
            }
        )
        return target
    if suffix == ["subjects", "kernel-input-distribution", "payload"]:
        target["panelId"] = "kernel-input-distribution"
        return target
    return None


def _register_managed_dictionary(request_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Use the current turn capability to freeze one Analyzer resource."""
    context = _managed_context()
    if context is None:
        return None
    backend_url = context.get("backend_url")
    capability_token = context.get("capability_token")
    if not isinstance(backend_url, str) or not isinstance(capability_token, str):
        raise AnalyzerToolError("managed Agent context is incomplete")
    request_body = json.dumps(
        request_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    registration_request = Request(
        urljoin(
            f"{backend_url.rstrip('/')}/",
            "api/internal/analyzer-citations/register",
        ),
        data=request_body,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {capability_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(registration_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            dictionary = json.loads(response.read(MAX_RESPONSE_BYTES + 1))
    except HTTPError as error:
        detail = error.read(2048).decode("utf-8", errors="replace")
        raise AnalyzerToolError(
            f"citation registration returned HTTP {error.code}: {detail}"
        ) from error
    except (URLError, json.JSONDecodeError) as error:
        raise AnalyzerToolError(f"citation registration failed: {error}") from error
    if not isinstance(dictionary, dict):
        raise AnalyzerToolError("citation registration returned an invalid dictionary")
    return dictionary


def _register_prediction_citations(
    safe_path: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    target = _prediction_target_from_path(safe_path)
    if target is None:
        return None
    dictionary = _register_managed_dictionary(
        {
            "resourceKind": "prediction",
            "predictionId": target["predictionId"],
            "resourcePath": safe_path,
        }
    )
    if dictionary is None:
        return None
    return dictionary, target


def _prediction_citation_token(
    dictionary: dict[str, Any], target: dict[str, Any]
) -> str:
    entries = dictionary.get("entries")
    if not isinstance(entries, list):
        raise AnalyzerToolError("prediction citation dictionary has no entries")
    matches = []
    for entry in entries:
        candidate = entry.get("target") if isinstance(entry, dict) else None
        if not isinstance(candidate, dict):
            continue
        if all(candidate.get(key) == value for key, value in target.items()):
            token = entry.get("token")
            if isinstance(token, str):
                matches.append(token)
    if len(matches) != 1:
        raise AnalyzerToolError(
            f"prediction citation dictionary has {len(matches)} matches for this resource"
        )
    return matches[0]


def _register_kernel_citations(
    safe_path: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Register exact profile/measurement JSON without exposing job identity."""
    segments = urlsplit(safe_path).path.strip("/").split("/")
    if len(segments) != 5 or segments[:2] != ["api", "v1"]:
        return None
    if segments[2] == "kernel-profiles" and segments[4] in {"descriptor", "curve"}:
        resource_kind = "kernel_profile"
        identifier_key = "profileId"
    elif segments[2] == "kernel-measurements" and segments[4] in {
        "descriptor",
        "summary",
    }:
        resource_kind = "kernel_measurement"
        identifier_key = "measurementId"
    else:
        return None
    dictionary = _register_managed_dictionary(
        {
            "resourceKind": resource_kind,
            identifier_key: segments[3],
            "resourcePath": safe_path,
            "analysis": payload,
        }
    )
    if dictionary is None:
        return None
    if not isinstance(dictionary.get("entries"), list):
        raise AnalyzerToolError("citation registration returned an invalid dictionary")
    citations: dict[str, str] = {}
    for entry in dictionary["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("target"), dict):
            continue
        token = entry.get("token")
        target = entry["target"]
        detail = target.get("metricKey") or target.get("plotName") or target.get("panelId")
        if isinstance(token, str) and isinstance(detail, str):
            citations[detail] = token
    return {"result": payload, "citations": citations}


def _register_run_citations(safe_path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    segments = urlsplit(safe_path).path.strip("/").split("/")
    if len(segments) < 5 or segments[:3] != ["api", "v1", "runs"]:
        return None
    dictionary = _register_managed_dictionary(
        {
            "resourceKind": "run",
            "runId": segments[3],
            "resourcePath": safe_path,
        }
    )
    if dictionary is None:
        return None
    entries = dictionary.get("entries")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise AnalyzerToolError("run citation registration returned an invalid dictionary")
    token = entries[0].get("token")
    if not isinstance(token, str):
        raise AnalyzerToolError("run citation registration returned no token")
    return {"citation": token, "result": payload}


def _unique_dictionary_token(
    entries: list[Any],
    *,
    experiment_id: str,
    metric_key: str,
    run_id: str | None,
) -> str:
    matches: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("target"), dict):
            continue
        target = entry["target"]
        if (
            target.get("kind") != "aggregate"
            or target.get("experimentId") != experiment_id
            or target.get("metricKey") != metric_key
        ):
            continue
        target_run_id = target.get("runId")
        if (run_id is None and target_run_id is not None) or (
            run_id is not None and target_run_id != run_id
        ):
            continue
        token = entry.get("token")
        if isinstance(token, str) and token:
            matches.append(token)
    if len(matches) != 1:
        scope = "panel" if run_id is None else f"run {run_id}"
        raise AnalyzerToolError(
            f"citation dictionary has {len(matches)} matches for {metric_key} at {scope}"
        )
    return matches[0]


def _compact_sweep_evidence(
    payload: dict[str, Any],
    dictionary: dict[str, Any],
) -> dict[str, Any]:
    """Project one exact sweep into a bounded, citation-adjacent evidence block."""
    axes = payload.get("axes")
    metrics = payload.get("metrics")
    runs = payload.get("runs")
    entries = dictionary.get("entries")
    experiment_id = payload.get("sweep_id")
    display_name = payload.get("display_name")
    if (
        payload.get("protocol_version") != 1
        or payload.get("schema_version") != 1
        or not isinstance(experiment_id, str)
        or not experiment_id
        or not isinstance(display_name, str)
        or not display_name
        or not isinstance(axes, list)
        or not all(isinstance(axis, str) and axis for axis in axes)
        or not isinstance(metrics, list)
        or not isinstance(runs, list)
        or not isinstance(entries, list)
    ):
        raise AnalyzerToolError("Analyzer sweep or citation dictionary is invalid")

    metric_projection: dict[str, Any] = {}
    metric_keys: list[str] = []
    for metric in metrics:
        if not isinstance(metric, dict):
            raise AnalyzerToolError("Analyzer sweep metric descriptor is invalid")
        metric_key = metric.get("key")
        label = metric.get("label")
        unit = metric.get("unit")
        objective = metric.get("objective")
        if (
            not isinstance(metric_key, str)
            or not metric_key
            or not isinstance(label, str)
            or not isinstance(unit, str)
            or objective not in {"minimize", "maximize"}
            or metric_key in metric_projection
        ):
            raise AnalyzerToolError("Analyzer sweep metric descriptor is invalid")
        metric_keys.append(metric_key)
        metric_projection[metric_key] = {
            "label": label,
            "unit": unit,
            "objective": objective,
            "citation": _unique_dictionary_token(
                entries,
                experiment_id=experiment_id,
                metric_key=metric_key,
                run_id=None,
            ),
        }

    row_projection: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("coordinates"), dict):
            raise AnalyzerToolError("Analyzer sweep row is invalid")
        run_id = run.get("run_id")
        raw_values = run.get("metrics")
        if (
            not isinstance(run_id, str)
            or not run_id
            or not isinstance(raw_values, dict)
        ):
            raise AnalyzerToolError(
                "Analyzer sweep row has no discoverable run identity"
            )
        if run_id in seen_run_ids:
            raise AnalyzerToolError("Analyzer sweep contains a duplicate run identity")
        seen_run_ids.add(run_id)
        coordinates = run["coordinates"]
        if any(axis not in coordinates for axis in axes):
            raise AnalyzerToolError("Analyzer sweep row is missing an axis coordinate")
        values: dict[str, Any] = {}
        for metric_key in metric_keys:
            if metric_key not in raw_values:
                raise AnalyzerToolError(
                    f"Analyzer sweep row is missing metric {metric_key}"
                )
            values[metric_key] = {
                "raw": raw_values[metric_key],
                "citation": _unique_dictionary_token(
                    entries,
                    experiment_id=experiment_id,
                    metric_key=metric_key,
                    run_id=run_id,
                ),
            }
        row_projection.append(
            {
                "coordinates": {axis: coordinates[axis] for axis in axes},
                "values": values,
            }
        )

    evidence = {
        "resource": {
            "id": experiment_id,
            "name": display_name,
        },
        "axes": axes,
        "metrics": metric_projection,
        "rows": row_projection,
    }
    encoded = json.dumps(
        evidence,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_EVIDENCE_BYTES:
        raise AnalyzerToolError(
            "compact Analyzer evidence exceeded the 1 MiB tool limit"
        )
    return evidence


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
        payload = json.loads(response_bytes)
    except json.JSONDecodeError as error:
        raise AnalyzerToolError("Analyzer returned invalid JSON") from error
    dictionary = _register_sweep_citations(safe_path, payload)
    if dictionary is not None:
        return _compact_sweep_evidence(payload, dictionary)
    prediction_registration = _register_prediction_citations(safe_path)
    if prediction_registration is not None:
        prediction_dictionary, prediction_target = prediction_registration
        return {
            "citation": _prediction_citation_token(
                prediction_dictionary, prediction_target
            ),
            "result": payload,
        }
    kernel_result = _register_kernel_citations(safe_path, payload)
    if kernel_result is not None:
        return kernel_result
    run_result = _register_run_citations(safe_path, payload)
    if run_result is not None:
        return run_result
    return payload


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
