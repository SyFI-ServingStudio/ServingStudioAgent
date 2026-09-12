"""Strict turn-time Analyzer context and Citation DSL v2 freezing.

The browser builds the initial dictionary for an already-open Analyzer surface.
For Agent-first turns, the managed Analyzer bridge may register a later sweep
dictionary after a simulation becomes discoverable. The conversation host
validates both bounded snapshots and freezes exact inline-code references
against the latest dictionary registered by that turn. V2 makes workspace
identity mandatory so an opaque resource cannot resolve in the wrong workspace.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

_CITATION_TOKEN = re.compile(
    r"^(?:exp|run|pred|kprof|kmeasure)\.[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$"
)
_INLINE_CODE = re.compile(r"(?<!`)`([^`\r\n]+)`(?!`)")
CoordinatePrimitive = str | int | float | bool | None
CoordinateValue = CoordinatePrimitive | list[CoordinatePrimitive]
_AXIS_ALIASES = {
    "tensor_parallel": "tp",
    "tp": "tp",
    "request_rate": "rate",
    "rate": "rate",
}
_MAX_AGGREGATE_ENTRIES = 1_800


class AggregateSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["aggregate"]
    workspace_id: str = Field(alias="workspaceId", min_length=1)
    experiment_id: str = Field(alias="experimentId", min_length=1)
    panel_id: str | None = Field(default=None, alias="panelId", min_length=1)
    metric_key: str | None = Field(default=None, alias="metricKey", min_length=1)
    statistic: Literal["mean", "p99"] | None = None
    run_id: str | None = Field(default=None, alias="runId", min_length=1)
    coordinates: dict[str, CoordinateValue] | None = None


class OperationRef(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    iter_id: str = Field(alias="iterId", min_length=1)
    batch_id: str = Field(alias="batchId", min_length=1)
    operation_id: str = Field(alias="operationId", min_length=1)


class RunSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["run"]
    workspace_id: str = Field(alias="workspaceId", min_length=1)
    run_id: str = Field(alias="runId", min_length=1)
    panel_id: str | None = Field(alias="panelId")
    scope: Literal["cluster", "pool", "worker", "kernel", "parallel"]
    pool_role: str | None = Field(alias="poolRole")
    worker_key: str | None = Field(alias="workerKey")
    leaf_id: int | None = Field(alias="leafId", ge=0)
    par_id: int | None = Field(alias="parId", ge=0)
    cursor_ms: float | None = Field(alias="cursorMs", ge=0)
    cursor_needs_seek: bool = Field(alias="cursorNeedsSeek")
    operation: OperationRef | None
    worker_analysis_level: Literal["worker", "iteration"] = Field(
        alias="workerAnalysisLevel"
    )


class PredictionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["prediction"]
    workspace_id: str = Field(alias="workspaceId", min_length=1)
    prediction_id: str = Field(alias="predictionId", min_length=1)
    panel_id: str | None = Field(alias="panelId")
    case_id: str | None = Field(alias="caseId", min_length=1)
    operation_id: str | None = Field(alias="operationId", min_length=1)
    leaf_id: int | None = Field(alias="leafId", ge=0)
    parallel_id: int | None = Field(alias="parallelId", ge=0)
    optimality_mode: Literal["unlocked", "batch_locked"] = Field(
        alias="optimalityMode"
    )


class KernelProfileSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["kernel_profile"]
    workspace_id: str = Field(alias="workspaceId", min_length=1)
    profile_id: str = Field(alias="profileId", min_length=1)
    panel_id: str | None = Field(alias="panelId")
    metric_key: str | None = Field(alias="metricKey", min_length=1)


class KernelMeasurementSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["kernel_measurement"]
    workspace_id: str = Field(alias="workspaceId", min_length=1)
    measurement_id: str = Field(alias="measurementId", min_length=1)
    panel_id: str | None = Field(alias="panelId")
    metric_key: str | None = Field(alias="metricKey", min_length=1)
    plot_name: str | None = Field(alias="plotName", min_length=1)


AnalyzerSelection = Annotated[
    AggregateSelection
    | RunSelection
    | PredictionSelection
    | KernelProfileSelection
    | KernelMeasurementSelection,
    Field(discriminator="kind"),
]


class AggregateEvidenceRef(AggregateSelection):
    protocol: Literal["vibesim.analyzer/v2"]


class RunEvidenceRef(RunSelection):
    protocol: Literal["vibesim.analyzer/v2"]


class PredictionEvidenceRef(PredictionSelection):
    protocol: Literal["vibesim.analyzer/v2"]


class KernelProfileEvidenceRef(KernelProfileSelection):
    protocol: Literal["vibesim.analyzer/v2"]


class KernelMeasurementEvidenceRef(KernelMeasurementSelection):
    protocol: Literal["vibesim.analyzer/v2"]


EvidenceRef = Annotated[
    AggregateEvidenceRef
    | RunEvidenceRef
    | PredictionEvidenceRef
    | KernelProfileEvidenceRef
    | KernelMeasurementEvidenceRef,
    Field(discriminator="kind"),
]


class CitationDictionaryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    token: str = Field(min_length=5, max_length=160)
    display_label: str = Field(alias="displayLabel", min_length=1, max_length=240)
    target: EvidenceRef

    @model_validator(mode="after")
    def validate_token_and_target(self) -> "CitationDictionaryEntry":
        if _CITATION_TOKEN.fullmatch(self.token) is None:
            raise ValueError("citation token does not match ServingStudioSim Citation DSL v2")
        return self


class CitationDictionarySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    protocol: Literal["vibesim.citation-dictionary/v2"]
    identity: str = Field(min_length=1, max_length=160)
    document: str = Field(min_length=1, max_length=64_000)
    entries: list[CitationDictionaryEntry] = Field(max_length=2_000)

    @model_validator(mode="after")
    def require_unique_tokens(self) -> "CitationDictionarySnapshot":
        tokens = [entry.token for entry in self.entries]
        if len(tokens) != len(set(tokens)):
            raise ValueError("citation dictionary tokens must be unique")
        return self


class AnalyzerTurnContext(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    protocol: Literal["vibesim.conversation-context/v2"]
    selection: AnalyzerSelection | None = None
    citation_dictionary: CitationDictionarySnapshot = Field(alias="citationDictionary")


class FrozenCitation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    protocol: Literal["vibesim.citation/v2"] = "vibesim.citation/v2"
    token: str
    source_start: int = Field(alias="sourceStart", ge=0)
    source_end: int = Field(alias="sourceEnd", ge=0)
    display_label: str = Field(alias="displayLabel")
    target: EvidenceRef


def _safe_segment(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_-")
    return normalized if normalized[:1].isalpha() else f"v{normalized or 'unknown'}"


def _compact_value(value: CoordinateValue) -> str:
    if isinstance(value, list):
        return "_".join(_compact_value(item) for item in value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value).replace("-", "m").replace(".", "p")
    return _safe_segment(value)


def _short_identity(value: str) -> str:
    """Match the frontend's unsigned FNV-1a base-36 identity."""
    hash_value = 0x811C9DC5
    for character in value:
        hash_value ^= ord(character)
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if hash_value == 0:
        return "0"
    encoded = ""
    while hash_value:
        hash_value, remainder = divmod(hash_value, 36)
        encoded = alphabet[remainder] + encoded
    return encoded


def _metric_statistic(metric: dict[str, Any]) -> Literal["mean", "p99"] | None:
    key = str(metric.get("key") or "")
    label = str(metric.get("label") or "")
    if re.search(r"(^|_)mean(_|$)", key) or re.match(r"^mean\b", label, re.I):
        return "mean"
    if re.search(r"(^|_)p99(_|$)", key) or re.match(r"^p99\b", label, re.I):
        return "p99"
    return None


def _metric_path(metric: dict[str, Any]) -> str:
    key = str(metric["key"])
    group = str(metric["group"])
    statistic = _metric_statistic(metric)
    if key == "total_tps":
        return "throughput"
    if group == "tpot" or "tpot" in key:
        return f"tpot.{statistic}" if statistic else "tpot"
    if group == "ttft" or "ttft" in key:
        return f"ttft.{statistic}" if statistic else "ttft"
    if group == "utilization" or "utilization" in key:
        return "utilization"
    return _safe_segment(key)


def build_aggregate_citation_dictionary(
    analysis: dict[str, Any],
    *,
    workspace_id: str,
    experiment_id: str,
) -> CitationDictionarySnapshot:
    """Build the same bounded aggregate DSL exposed by the browser.

    ``workspace_id`` and ``experiment_id`` come from the managed capability and
    registry, never from the workspace-local Analyzer payload. A local Analyzer
    uses an implementation workspace id that is not a UI navigation identity.
    """
    if analysis.get("protocol_version") != 1 or analysis.get("schema_version") != 1:
        raise ValueError("unsupported Analyzer sweep payload version")
    if str(analysis.get("sweep_id") or "") != experiment_id:
        raise ValueError("Analyzer sweep identity does not match managed experiment")
    axes = analysis.get("axes")
    metrics = analysis.get("metrics")
    runs = analysis.get("runs")
    if not isinstance(axes, list) or not all(
        isinstance(axis, str) and axis for axis in axes
    ):
        raise ValueError("Analyzer sweep axes are invalid")
    if not isinstance(metrics, list) or not isinstance(runs, list):
        raise ValueError("Analyzer sweep metrics or runs are invalid")

    used_aliases: set[str] = set()
    axis_dictionaries: list[
        tuple[str, str, dict[str, tuple[str, CoordinateValue]]]
    ] = []
    for axis in axes:
        preferred = _AXIS_ALIASES.get(axis, _safe_segment(axis))
        alias = preferred
        suffix = 2
        while alias in used_aliases:
            alias = f"{preferred}{suffix}"
            suffix += 1
        used_aliases.add(alias)
        values: dict[str, tuple[str, CoordinateValue]] = {}
        used_segments: dict[str, str] = {}
        for run in runs:
            if not isinstance(run, dict) or not isinstance(
                run.get("coordinates"), dict
            ):
                raise ValueError("Analyzer sweep run is invalid")
            if axis not in run["coordinates"]:
                continue
            value = run["coordinates"][axis]
            identity = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if identity in values:
                continue
            base_segment = f"{alias}{_compact_value(value)}"
            existing_identity = used_segments.get(base_segment)
            segment = (
                base_segment
                if existing_identity in {None, identity}
                else f"{base_segment}_{_short_identity(identity)}"
            )
            used_segments[segment] = identity
            values[identity] = (segment, value)
        axis_dictionaries.append((axis, alias, values))

    unique_metrics: list[tuple[dict[str, Any], str, str]] = []
    seen_paths: set[str] = set()
    for raw_metric in metrics:
        if not isinstance(raw_metric, dict):
            raise ValueError("Analyzer sweep metric is invalid")
        required = ("key", "label", "group", "unit")
        if any(not isinstance(raw_metric.get(field), str) for field in required):
            raise ValueError("Analyzer sweep metric descriptor is invalid")
        path = _metric_path(raw_metric)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        statistic = _metric_statistic(raw_metric)
        panel_id = str(raw_metric["group"]) if statistic else str(raw_metric["key"])
        unique_metrics.append((raw_metric, path, panel_id))

    entries: list[dict[str, Any]] = []
    for metric, path, panel_id in unique_metrics:
        target: dict[str, Any] = {
            "protocol": "vibesim.analyzer/v2",
            "kind": "aggregate",
            "workspaceId": workspace_id,
            "experimentId": experiment_id,
            "panelId": panel_id,
            "metricKey": metric["key"],
        }
        statistic = _metric_statistic(metric)
        if statistic:
            target["statistic"] = statistic
        entries.append(
            {
                "token": f"exp.{path}",
                "displayLabel": f"{metric['label']} · all coordinates",
                "target": target,
            }
        )

    member_rows: list[str] = []
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("coordinates"), dict):
            raise ValueError("Analyzer sweep run is invalid")
        # A zero-axis aggregate is a singleton: its panel citation already
        # identifies the only run, and there is no coordinate segment to add.
        if not axis_dictionaries:
            continue
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            continue
        coordinates = run["coordinates"]
        segments: list[str] = []
        for axis, _alias, values in axis_dictionaries:
            if axis not in coordinates:
                segments = []
                break
            identity = json.dumps(
                coordinates[axis], ensure_ascii=False, separators=(",", ":")
            )
            dictionary_value = values.get(identity)
            if dictionary_value is None:
                segments = []
                break
            segments.append(dictionary_value[0])
        if len(segments) != len(axis_dictionaries):
            continue
        prefix = ".".join(segments)
        member_rows.append(prefix)
        coordinate_values = {axis: coordinates[axis] for axis in axes}
        labels = run.get("labels") if isinstance(run.get("labels"), dict) else {}
        coordinate_label = " · ".join(
            str(labels.get(axis) or f"{axis}={coordinates[axis]}") for axis in axes
        )
        for metric, path, panel_id in unique_metrics:
            if len(entries) >= _MAX_AGGREGATE_ENTRIES:
                break
            target = {
                "protocol": "vibesim.analyzer/v2",
                "kind": "aggregate",
                "workspaceId": workspace_id,
                "experimentId": experiment_id,
                "panelId": panel_id,
                "metricKey": metric["key"],
                "runId": run_id,
                "coordinates": coordinate_values,
            }
            statistic = _metric_statistic(metric)
            if statistic:
                target["statistic"] = statistic
            entries.append(
                {
                    "token": f"exp.{prefix}.{path}",
                    "displayLabel": f"{coordinate_label} · {metric['label']}",
                    "target": target,
                }
            )

    axis_rows = [
        "  - `{}<value>`: {}; valid segments: {}".format(
            alias,
            axis,
            ", ".join(f"`{segment}`" for segment, _value in values.values()),
        )
        for axis, alias, values in axis_dictionaries
    ]
    metric_rows = [
        f"  - `{path}`: {metric['label']}, {metric['unit']}"
        for metric, path, _panel_id in unique_metrics
    ]
    document = "\n".join(
        [
            "## Analyzer citation references",
            "",
            "Use only these exact Markdown inline-code references. Citation text never navigates until the user clicks it.",
            "",
            "Experiment `exp`",
            "- axes, in launcher declaration order:",
            *(axis_rows or ["  - none (single configuration)"]),
            "- valid members:",
            *(f"  - `{member}`" for member in member_rows),
            "- metrics:",
            *metric_rows,
            (
                "- forms: `exp.<metric>` and `exp.<member>.<metric>`"
                if axis_dictionaries
                else "- form: `exp.<metric>`"
            ),
        ]
    )
    identity_source = "|".join(
        f"{entry['token']}:{entry['target'].get('runId', '')}" for entry in entries
    )
    return CitationDictionarySnapshot.model_validate(
        {
            "protocol": "vibesim.citation-dictionary/v2",
            "identity": f"aggregate-{_short_identity(f'{experiment_id}|{identity_source}')}",
            "document": document,
            "entries": entries,
        }
    )


def _citation_path_segments(resource_path: str) -> list[str]:
    """Normalize current subject endpoints for the legacy-compatible target builders."""
    segments = urlsplit(resource_path).path.strip("/").split("/")
    if segments[:3] != ["api", "analyzer", "v1"]:
        return segments
    segments = ["api", "v1", *segments[3:]]
    if len(segments) < 5:
        return segments
    collection, suffix = segments[2], segments[4:]
    if collection in {"predictions", "kernel-profiles", "kernel-measurements"} or (
        collection == "runs" and suffix[0] == "workers"
    ):
        if (
            len(suffix) >= 3
            and suffix[-3] == "subjects"
            and suffix[-1] in {"payload", "report"}
        ):
            subject = suffix[-2]
            if collection == "predictions" and subject == "kernel-input-distribution":
                return segments
            suffix = [*suffix[:-3], subject]
            if (
                subject == "kernel-throughput-analysis"
                and len(suffix) >= 3
                and suffix[-3] == "leaves"
            ):
                suffix[-3] = "cost-tree"
    return [*segments[:4], *suffix]


def build_prediction_citation_dictionary(
    *,
    workspace_id: str,
    prediction_id: str,
    resource_path: str,
) -> CitationDictionarySnapshot:
    """Build one exact, path-bound timing-prediction citation."""
    parsed_resource = urlsplit(resource_path)
    query = parse_qs(parsed_resource.query)
    optimality_mode = query.get("mode", ["unlocked"])[0]
    if optimality_mode not in {"unlocked", "batch_locked"}:
        raise ValueError("prediction citation path has an invalid optimality mode")
    segments = _citation_path_segments(resource_path)
    expected_prefix = ["api", "v1", "predictions", prediction_id]
    if segments[:4] != expected_prefix:
        raise ValueError("prediction citation path does not match its resource id")

    case_id: str | None = None
    operation_id: str | None = None
    leaf_id: int | None = None
    panel_id: str | None = None
    suffix = segments[4:]
    if suffix == ["descriptor"] or suffix == ["cases"]:
        pass
    elif len(suffix) == 3 and suffix[0] == "cases" and suffix[2] in {
        "optimality-waterfall",
        "optimality-kernel-ladder",
    }:
        case_id = suffix[1]
        panel_id = (
            "optimality-breakdown"
            if suffix[2] == "optimality-waterfall"
            else "optimality-kernel-ladder"
        )
    elif (
        len(suffix) == 5
        and suffix[0] == "cases"
        and suffix[2] == "operations"
        and suffix[4] == "cost-tree"
    ):
        case_id, operation_id = suffix[1], suffix[3]
        panel_id = "cost-tree"
    elif (
        len(suffix) == 7
        and suffix[0] == "cases"
        and suffix[2] == "operations"
        and suffix[4] == "cost-tree"
        and suffix[6] == "kernel-throughput-analysis"
    ):
        case_id, operation_id = suffix[1], suffix[3]
        try:
            leaf_id = int(suffix[5])
        except ValueError as error:
            raise ValueError("prediction kernel citation has an invalid leaf id") from error
        if leaf_id < 0:
            raise ValueError("prediction kernel citation has an invalid leaf id")
        panel_id = "kernel-throughput"
    elif suffix == ["subjects", "kernel-input-distribution", "payload"]:
        panel_id = "kernel-input-distribution"
    else:
        raise ValueError("unsupported prediction citation resource path")

    token_segments = ["pred"]
    if case_id is not None:
        token_segments.append(f"case{_safe_segment(case_id)}")
    if operation_id is not None:
        token_segments.append(f"operation{_safe_segment(operation_id)}")
    if leaf_id is not None:
        token_segments.append(f"kernel{leaf_id}")
    if optimality_mode == "batch_locked":
        token_segments.append("batch_locked")
    panel_token = panel_id or "overview"
    token_segments.append(_safe_segment(panel_token))
    token = ".".join(token_segments)
    target = {
        "protocol": "vibesim.analyzer/v2",
        "kind": "prediction",
        "workspaceId": workspace_id,
        "predictionId": prediction_id,
        "panelId": panel_id,
        "caseId": case_id,
        "operationId": operation_id,
        "leafId": leaf_id,
        "parallelId": None,
        "optimalityMode": optimality_mode,
    }
    label_parts = [f"prediction {prediction_id}"]
    if case_id is not None:
        label_parts.append(f"case {case_id}")
    if operation_id is not None:
        label_parts.append(f"operation {operation_id}")
    if leaf_id is not None:
        label_parts.append(f"kernel {leaf_id}")
    label_parts.append(panel_token)
    display_label = " · ".join(label_parts)
    return CitationDictionarySnapshot.model_validate(
        {
            "protocol": "vibesim.citation-dictionary/v2",
            "identity": f"prediction-{_short_identity(f'{prediction_id}|{token}')}",
            "document": f"Use `{token}` for {display_label}.",
            "entries": [
                {
                    "token": token,
                    "displayLabel": display_label,
                    "target": target,
                }
            ],
        }
    )


def build_run_citation_dictionary(
    *, workspace_id: str, run_id: str, resource_path: str
) -> CitationDictionarySnapshot:
    """Bind one exact Analyzer run endpoint to a navigable run selection."""
    segments = _citation_path_segments(resource_path)
    if segments[:4] != ["api", "v1", "runs", run_id]:
        raise ValueError("run citation path does not match its resource id")
    suffix = segments[4:]
    scope = "cluster"
    panel_id = "overview"
    pool_role = None
    worker_key = None
    leaf_id = None
    operation = None
    subject_panels = {
        "throughput": "throughput",
        "utilization": "utilization",
        "request-state": "request-state",
        "kv-occupancy": "kv-cache",
        "kernel-time-share": "kernel-time-breakdown",
        "optimality": "optimality-breakdown",
        "kernel-input-distribution": "kernel-input-distribution",
    }
    if len(suffix) >= 3 and suffix[0] == "subjects":
        panel_id = subject_panels.get(suffix[1], _safe_segment(suffix[1]))
    elif len(suffix) >= 3 and suffix[0] == "workers":
        scope = "worker"
        pool_role, worker_key = suffix[1], f"{suffix[1]}/{suffix[2]}"
        if "cost-tree" in suffix:
            panel_id = "cost-tree"
            cost_tree_index = suffix.index("cost-tree")
            if cost_tree_index >= 6:
                operation = {
                    "iterId": suffix[cost_tree_index - 3],
                    "batchId": suffix[cost_tree_index - 2],
                    "operationId": suffix[cost_tree_index - 1],
                }
        elif "operations" in suffix:
            panel_id = "operation-timeline"
        if suffix[-1] == "kernel-throughput-analysis" and len(suffix) >= 2:
            try:
                leaf_id = int(suffix[-2])
            except ValueError as error:
                raise ValueError("run kernel citation has an invalid leaf id") from error
            scope = "kernel"
            panel_id = "kernel-throughput"
    elif suffix and suffix[0] not in {"descriptor", "summary", "topology", "model", "workload"}:
        panel_id = _safe_segment(suffix[-1])
    token = f"run.{scope}.{_safe_segment(panel_id)}"
    target = {
        "protocol": "vibesim.analyzer/v2",
        "kind": "run",
        "workspaceId": workspace_id,
        "runId": run_id,
        "panelId": panel_id,
        "scope": scope,
        "poolRole": pool_role,
        "workerKey": worker_key,
        "leafId": leaf_id,
        "parId": None,
        "cursorMs": None,
        "cursorNeedsSeek": False,
        "operation": operation,
        "workerAnalysisLevel": "iteration" if operation else "worker",
    }
    return CitationDictionarySnapshot.model_validate(
        {
            "protocol": "vibesim.citation-dictionary/v2",
            "identity": f"run-{_short_identity(f'{run_id}|{resource_path}')}",
            "document": f"Use `{token}` for run {run_id} · {panel_id}.",
            "entries": [
                {
                    "token": token,
                    "displayLabel": f"run {run_id} · {panel_id}",
                    "target": target,
                }
            ],
        }
    )


def build_kernel_profile_citation_dictionary(
    *,
    workspace_id: str,
    profile_id: str,
    resource_path: str,
    analysis: dict[str, Any],
) -> CitationDictionarySnapshot:
    """Bind a profile descriptor or each curve metric to its first-class page."""
    segments = _citation_path_segments(resource_path)
    if segments[:4] != ["api", "v1", "kernel-profiles", profile_id]:
        raise ValueError("kernel profile path does not match its resource id")
    suffix = segments[4:]
    if suffix == ["descriptor"]:
        metrics: list[str | None] = [None]
        panel_id = "overview"
    elif suffix == ["curve"]:
        raw_series = analysis.get("series")
        metrics = (
            [
                str(series["metric"])
                for series in raw_series
                if isinstance(series, dict) and series.get("metric")
            ]
            if isinstance(raw_series, list)
            else []
        ) or [None]
        panel_id = "curve"
    else:
        raise ValueError("unsupported kernel profile citation resource path")
    entries = []
    for metric_key in metrics:
        token = ".".join(
            ["kprof", panel_id, *([_safe_segment(metric_key)] if metric_key else [])]
        )
        entries.append(
            {
                "token": token,
                "displayLabel": " · ".join(
                    part for part in [profile_id, panel_id, metric_key] if part
                ),
                "target": {
                    "protocol": "vibesim.analyzer/v2",
                    "kind": "kernel_profile",
                    "workspaceId": workspace_id,
                    "profileId": profile_id,
                    "panelId": panel_id,
                    "metricKey": metric_key,
                },
            }
        )
    return CitationDictionarySnapshot.model_validate(
        {
            "protocol": "vibesim.citation-dictionary/v2",
            "identity": f"kernel-profile-{_short_identity(f'{profile_id}|{resource_path}')}",
            "document": "\n".join(
                f"Use `{entry['token']}` for {entry['displayLabel']}." for entry in entries
            ),
            "entries": entries,
        }
    )


def build_kernel_measurement_citation_dictionary(
    *,
    workspace_id: str,
    measurement_id: str,
    resource_path: str,
    analysis: dict[str, Any],
) -> CitationDictionarySnapshot:
    """Bind measurement summary values and declared plots to exact UI cards."""
    segments = _citation_path_segments(resource_path)
    if segments[:4] != ["api", "v1", "kernel-measurements", measurement_id]:
        raise ValueError("kernel measurement path does not match its resource id")
    suffix = segments[4:]
    targets: list[tuple[str, str | None, str | None]] = []
    if suffix == ["summary"]:
        runtime = analysis.get("runtime_ms", analysis.get("runtimeMs"))
        if isinstance(runtime, dict):
            targets.extend(("summary", str(metric), None) for metric in runtime)
        if not targets:
            targets.append(("summary", None, None))
    elif suffix == ["descriptor"]:
        targets.append(("overview", None, None))
        plot_urls = analysis.get("plot_urls", analysis.get("plotUrls"))
        if isinstance(plot_urls, list):
            for plot_url in plot_urls:
                if isinstance(plot_url, str) and plot_url:
                    targets.append(
                        ("plot", None, urlsplit(plot_url).path.rsplit("/", 1)[-1])
                    )
    else:
        raise ValueError("unsupported kernel measurement citation resource path")
    entries = []
    for panel_id, metric_key, plot_name in targets:
        detail = metric_key or plot_name
        token = ".".join(
            ["kmeasure", panel_id, *([_safe_segment(detail)] if detail else [])]
        )
        entries.append(
            {
                "token": token,
                "displayLabel": " · ".join(
                    part for part in [measurement_id, panel_id, detail] if part
                ),
                "target": {
                    "protocol": "vibesim.analyzer/v2",
                    "kind": "kernel_measurement",
                    "workspaceId": workspace_id,
                    "measurementId": measurement_id,
                    "panelId": panel_id,
                    "metricKey": metric_key,
                    "plotName": plot_name,
                },
            }
        )
    return CitationDictionarySnapshot.model_validate(
        {
            "protocol": "vibesim.citation-dictionary/v2",
            "identity": (
                "kernel-measurement-"
                f"{_short_identity(f'{measurement_id}|{resource_path}')}"
            ),
            "document": "\n".join(
                f"Use `{entry['token']}` for {entry['displayLabel']}." for entry in entries
            ),
            "entries": entries,
        }
    )


def latest_citation_dictionary(
    events: list[dict[str, Any]],
    initial: CitationDictionarySnapshot | None = None,
) -> CitationDictionarySnapshot | None:
    for event in reversed(events):
        if event["kind"] == "citation.dictionary":
            return CitationDictionarySnapshot.model_validate(event["payload"]["dictionary"])
    return initial


def merge_citation_dictionaries(
    previous: CitationDictionarySnapshot | None,
    current: CitationDictionarySnapshot,
) -> CitationDictionarySnapshot:
    """Accumulate exact resources read during one turn without duplicating tokens."""
    if previous is None:
        return current
    entries_by_token = {entry.token: entry for entry in previous.entries}
    entries_by_token.update({entry.token: entry for entry in current.entries})
    entries = list(entries_by_token.values())
    identity_source = "|".join(entry.token for entry in entries)
    return CitationDictionarySnapshot.model_validate(
        {
            "protocol": "vibesim.citation-dictionary/v2",
            "identity": f"combined-{_short_identity(identity_source)}",
            "document": "\n\n".join([previous.document, current.document]),
            "entries": [
                # Nullable selector fields are part of the strict evidence-ref
                # shape. Preserve explicit nulls while combining dictionaries;
                # dropping them turns a valid target into an incomplete one.
                entry.model_dump(by_alias=True) for entry in entries
            ],
        }
    )


def prompt_with_analyzer_context(text: str, context: AnalyzerTurnContext | None) -> str:
    """Attach only the active selection; MCP supplies current evidence tokens."""
    if context is None:
        return text
    selection_json = json.dumps(
        (
            context.selection.model_dump(by_alias=True)
            if context.selection is not None
            else None
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{text}\n\n"
        "## Active Analyzer context\n"
        f"Selection (literal JSON): `{selection_json}`\n\n"
        "Read this exact resource through the Analyzer MCP before reporting its "
        "values. Exact sweep, run, prediction, profile, and measurement "
        "responses place complete "
        "citation token beside the returned evidence. Copy the matching token "
        "unchanged as Markdown inline code; "
        "do not invent tokens or write navigation JSON. A citation navigates only "
        "after the user clicks it."
    )


def freeze_citations(
    markdown: str,
    dictionary: CitationDictionarySnapshot | None,
) -> list[dict[str, Any]]:
    """Freeze exact single-backtick inline-code references from one assistant turn."""
    if dictionary is None:
        return []
    entries_by_token = {entry.token: entry for entry in dictionary.entries}
    frozen: list[dict[str, Any]] = []
    for match in _INLINE_CODE.finditer(markdown):
        token = match.group(1)
        entry = entries_by_token.get(token)
        if entry is None:
            continue
        frozen.append(
            FrozenCitation(
                token=token,
                sourceStart=match.start(),
                sourceEnd=match.end(),
                displayLabel=entry.display_label,
                target=entry.target,
            ).model_dump(by_alias=True, exclude_none=True)
        )
    return frozen


def persisted_context(context: AnalyzerTurnContext | None) -> dict[str, Any] | None:
    return context.model_dump(by_alias=True) if context is not None else None
