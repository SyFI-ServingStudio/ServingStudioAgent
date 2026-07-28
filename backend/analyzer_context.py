"""Strict turn-time Analyzer context and Citation DSL v2 freezing.

The browser owns dictionary construction because it knows the active launcher
axes and registered evidence panels. The conversation host only validates the
bounded snapshot, exposes it to Codex, and freezes exact inline-code references
against that immutable allowlist. V2 makes the workspace identity mandatory so
an opaque Analyzer resource can never be resolved in the wrong workspace.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_CITATION_TOKEN = re.compile(r"^(?:exp|run)\.[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$")
_INLINE_CODE = re.compile(r"(?<!`)`([^`\r\n]+)`(?!`)")
CoordinatePrimitive = str | int | float | bool | None
CoordinateValue = CoordinatePrimitive | list[CoordinatePrimitive]


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
