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

from pydantic import BaseModel, ConfigDict, Field, model_validator

_CITATION_TOKEN = re.compile(r"^(?:exp|run)\.[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$")
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


AnalyzerSelection = Annotated[
    AggregateSelection | RunSelection,
    Field(discriminator="kind"),
]


class AggregateEvidenceRef(AggregateSelection):
    protocol: Literal["vibesim.analyzer/v2"]


class RunEvidenceRef(RunSelection):
    protocol: Literal["vibesim.analyzer/v2"]


EvidenceRef = Annotated[
    AggregateEvidenceRef | RunEvidenceRef,
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
            raise ValueError("citation token does not match VibeSim Citation DSL v2")
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
    if not isinstance(axes, list) or not all(isinstance(axis, str) and axis for axis in axes):
        raise ValueError("Analyzer sweep axes are invalid")
    if not isinstance(metrics, list) or not isinstance(runs, list):
        raise ValueError("Analyzer sweep metrics or runs are invalid")

    used_aliases: set[str] = set()
    axis_dictionaries: list[tuple[str, str, dict[str, tuple[str, CoordinateValue]]]] = []
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
            if not isinstance(run, dict) or not isinstance(run.get("coordinates"), dict):
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
            "- forms: `exp.<metric>` and `exp.<member>.<metric>`",
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


def prompt_with_analyzer_context(text: str, context: AnalyzerTurnContext | None) -> str:
    """Attach a bounded machine snapshot without changing stored user prose."""
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
        f"{context.citation_dictionary.document.strip()}\n\n"
        "When a claim is supported by Analyzer evidence, cite only an exact token "
        "listed above as Markdown inline code. Do not invent tokens, write Analyzer "
        "URLs, opaque ids, percent encoding, or JSON citations. Writing a citation "
        "does not navigate the Analyzer; the user decides whether to open it."
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
