from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

from vibesim_agent.domain.evidence import (
    AnalyzerTurnContext,
    CitationDictionarySnapshot,
    build_aggregate_citation_dictionary,
    build_kernel_measurement_citation_dictionary,
    build_kernel_profile_citation_dictionary,
    build_prediction_citation_dictionary,
    build_run_citation_dictionary,
    freeze_citations,
    merge_citation_dictionaries,
    persisted_context,
    prompt_with_analyzer_context,
)


def dictionary() -> CitationDictionarySnapshot:
    return CitationDictionarySnapshot.model_validate(
        {
            "protocol": "vibesim.citation-dictionary/v2",
            "identity": "s_test:1",
            "document": "Use `exp.tp2.rate20.throughput`.",
            "entries": [
                {
                    "token": "exp.tp2.rate20.throughput",
                    "displayLabel": "TP=2 · rate=20 · Total throughput",
                    "target": {
                        "protocol": "vibesim.analyzer/v2",
                        "kind": "aggregate",
                        "workspaceId": "w_test",
                        "experimentId": "s_test",
                        "panelId": "total_tps",
                        "metricKey": "total_tps",
                        "runId": "r_test",
                        "coordinates": {"tensor_parallel": 2, "request_rate": 20},
                    },
                }
            ],
        }
    )


class AnalyzerContextTests(unittest.TestCase):
    def test_domain_import_does_not_load_backend(self):
        script = """
import importlib.abc
import sys
class RejectBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "backend" or fullname.startswith("backend."):
            raise AssertionError("legacy backend import: " + fullname)
sys.meta_path.insert(0, RejectBackend())
import vibesim_agent.domain.evidence
assert not any(name == "backend" or name.startswith("backend.") for name in sys.modules)
"""
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_persisted_context_schema_prompt_and_freezing_match_legacy_bytes(self):
        golden = json.loads((Path(__file__).parent / "fixtures/legacy_analyzer_context.json").read_text())

        payload = {
            "protocol": "vibesim.conversation-context/v2",
            "selection": None,
            "citationDictionary": dictionary().model_dump(by_alias=True),
        }
        context = AnalyzerTurnContext.model_validate(payload)
        self.assertEqual(payload, golden["payload"])
        self.assertEqual(AnalyzerTurnContext.model_json_schema(), golden["schema"])
        self.assertEqual(json.dumps(persisted_context(context), ensure_ascii=False),
                         golden["persisted_json"])
        markdown = "Use `exp.tp2.rate20.throughput` twice: `exp.tp2.rate20.throughput`."
        self.assertEqual(prompt_with_analyzer_context(markdown, context).encode(),
                         golden["prompt"].encode())
        self.assertEqual(json.dumps(freeze_citations(markdown, context.citation_dictionary), ensure_ascii=False),
                         golden["frozen_json"])
        self.assertIsNone(persisted_context(None))
        self.assertEqual(freeze_citations(markdown, None), [])
        self.assertEqual(prompt_with_analyzer_context(markdown, None), markdown)

    def test_dictionary_merge_preserves_nullable_selectors_and_latest_target(self):
        golden = json.loads((Path(__file__).parent / "fixtures/legacy_analyzer_context.json").read_text())

        previous = build_run_citation_dictionary(
            workspace_id="w_test", run_id="old",
            resource_path="/api/v1/runs/old/subjects/utilization/payload",
        )
        current = build_run_citation_dictionary(
            workspace_id="w_test", run_id="new",
            resource_path="/api/v1/runs/new/subjects/utilization/payload",
        )
        merged = merge_citation_dictionaries(previous, current)
        self.assertEqual(previous.model_dump_json(by_alias=True), golden["previous_json"])
        self.assertEqual(current.model_dump_json(by_alias=True), golden["current_json"])
        self.assertEqual(merged.model_dump_json(by_alias=True), golden["merged_json"])
        self.assertEqual(len(merged.entries), 1)
        self.assertEqual(merged.entries[0].target.run_id, "new")
        self.assertIn('"poolRole":null', merged.model_dump_json(by_alias=True))

    def test_builds_managed_aggregate_dictionary_with_authoritative_identity(
        self,
    ) -> None:
        snapshot = build_aggregate_citation_dictionary(
            {
                "protocol_version": 1,
                "schema_version": 1,
                "workspace_id": "root_0",
                "sweep_id": "e_test",
                "axes": ["tensor_parallel", "request_rate"],
                "metrics": [
                    {
                        "key": "total_tps",
                        "label": "Total throughput",
                        "group": "throughput",
                        "unit": "token/s",
                        "objective": "maximize",
                    },
                    {
                        "key": "mean_ttft_ms",
                        "label": "Mean TTFT",
                        "group": "ttft",
                        "unit": "ms",
                        "objective": "minimize",
                    },
                ],
                "runs": [
                    {
                        "run_id": "r_test",
                        "coordinates": {
                            "tensor_parallel": 2,
                            "request_rate": 20,
                        },
                        "labels": {
                            "tensor_parallel": "TP 2",
                            "request_rate": "20 req/s",
                        },
                    }
                ],
            },
            workspace_id="w_managed",
            experiment_id="e_test",
        )

        entries = {entry.token: entry for entry in snapshot.entries}
        self.assertIn("exp.throughput", entries)
        self.assertIn("exp.tp2.rate20.throughput", entries)
        self.assertIn("exp.tp2.rate20.ttft.mean", entries)
        target = entries["exp.tp2.rate20.throughput"].target
        self.assertEqual(target.workspace_id, "w_managed")
        self.assertEqual(target.experiment_id, "e_test")
        self.assertEqual(target.run_id, "r_test")

    def test_builds_singleton_dictionary_without_empty_coordinate_token(self) -> None:
        snapshot = build_aggregate_citation_dictionary(
            {
                "protocol_version": 1,
                "schema_version": 1,
                "sweep_id": "e_singleton",
                "axes": [],
                "metrics": [
                    {
                        "key": "total_tps",
                        "label": "Total throughput",
                        "group": "throughput",
                        "unit": "token/s",
                        "objective": "maximize",
                    }
                ],
                "runs": [
                    {
                        "run_id": "r_singleton",
                        "coordinates": {},
                        "labels": {},
                    }
                ],
            },
            workspace_id="w_managed",
            experiment_id="e_singleton",
        )

        self.assertEqual(
            [entry.token for entry in snapshot.entries],
            ["exp.throughput"],
        )
        target = snapshot.entries[0].target
        self.assertEqual(target.experiment_id, "e_singleton")
        self.assertIsNone(target.run_id)
        self.assertIn("- form: `exp.<metric>`", snapshot.document)
        self.assertNotIn("exp..", snapshot.document)

    def test_freeze_citations_only_accepts_exact_inline_allowlist_tokens(self) -> None:
        markdown = (
            "Prose exp.tp2.rate20.throughput is not linked. "
            "Valid `exp.tp2.rate20.throughput`; invented `exp.tp4.rate20.throughput`."
        )
        citations = freeze_citations(markdown, dictionary())

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["token"], "exp.tp2.rate20.throughput")
        self.assertEqual(
            markdown[citations[0]["sourceStart"] : citations[0]["sourceEnd"]],
            "`exp.tp2.rate20.throughput`",
        )
        self.assertEqual(citations[0]["target"]["runId"], "r_test")
        self.assertNotIn("statistic", citations[0]["target"])

    def test_builds_exact_prediction_dictionary_from_analyzer_path(self) -> None:
        snapshot = build_prediction_citation_dictionary(
            workspace_id="w_managed",
            prediction_id="p_test",
            resource_path=(
                "/api/v1/predictions/p_test/cases/40/"
                "optimality-waterfall?mode=batch_locked"
            ),
        )

        self.assertEqual(len(snapshot.entries), 1)
        entry = snapshot.entries[0]
        self.assertEqual(
            entry.token,
            "pred.casev40.batch_locked.optimality-breakdown",
        )
        self.assertEqual(entry.target.kind, "prediction")
        self.assertEqual(entry.target.prediction_id, "p_test")
        self.assertEqual(entry.target.case_id, "40")
        self.assertEqual(entry.target.panel_id, "optimality-breakdown")
        self.assertEqual(entry.target.optimality_mode, "batch_locked")
        self.assertFalse(hasattr(entry.target, "run_id"))

        citations = freeze_citations(
            f"Evidence `{entry.token}`.",
            snapshot,
        )
        self.assertEqual(citations[0]["target"]["predictionId"], "p_test")
        self.assertNotIn("runId", citations[0]["target"])

    def test_builds_run_and_kernel_resource_dictionaries(self) -> None:
        run = build_run_citation_dictionary(
            workspace_id="w_managed",
            run_id="r_test",
            resource_path="/api/v1/runs/r_test/subjects/utilization/payload",
        )
        profile = build_kernel_profile_citation_dictionary(
            workspace_id="w_managed",
            profile_id="kp_test",
            resource_path="/api/v1/kernel-profiles/kp_test/curve",
            analysis={"series": [{"metric": "time_ms"}, {"metric": "tflops"}]},
        )
        measurement = build_kernel_measurement_citation_dictionary(
            workspace_id="w_managed",
            measurement_id="km_test",
            resource_path="/api/v1/kernel-measurements/km_test/summary",
            analysis={"runtime_ms": {"median": 1.2, "p99": 1.8}},
        )

        self.assertEqual(run.entries[0].token, "run.cluster.utilization")
        self.assertEqual(
            [entry.token for entry in profile.entries],
            ["kprof.curve.time_ms", "kprof.curve.tflops"],
        )
        self.assertEqual(
            [entry.token for entry in measurement.entries],
            ["kmeasure.summary.median", "kmeasure.summary.p99"],
        )

    def test_dictionary_rejects_duplicate_or_non_dsl_tokens(self) -> None:
        payload = dictionary().model_dump(by_alias=True)
        payload["entries"].append(dict(payload["entries"][0]))
        with self.assertRaisesRegex(ValidationError, "tokens must be unique"):
            CitationDictionarySnapshot.model_validate(payload)

        payload["entries"] = [
            {
                **payload["entries"][0],
                "token": "https://example.test/evidence",
            }
        ]
        with self.assertRaisesRegex(ValidationError, "Citation DSL"):
            CitationDictionarySnapshot.model_validate(payload)

    def test_dictionary_rejects_unbounded_or_incomplete_targets(self) -> None:
        payload = dictionary().model_dump(by_alias=True)
        payload["entries"][0]["target"]["selector"] = ".chart"
        with self.assertRaises(ValidationError):
            CitationDictionarySnapshot.model_validate(payload)

        payload = dictionary().model_dump(by_alias=True)
        payload["entries"][0]["target"] = {
            "protocol": "vibesim.analyzer/v2",
            "kind": "run",
            "workspaceId": "w_test",
            "runId": "r_test",
            "scope": "worker",
        }
        with self.assertRaises(ValidationError):
            CitationDictionarySnapshot.model_validate(payload)

    def test_prompt_exposes_selection_without_the_verbose_dictionary(self) -> None:
        context = AnalyzerTurnContext.model_validate(
            {
                "protocol": "vibesim.conversation-context/v2",
                "selection": {
                    "kind": "aggregate",
                    "workspaceId": "w_test",
                    "experimentId": "s_test",
                    "panelId": "total_tps",
                },
                "citationDictionary": dictionary().model_dump(by_alias=True),
            }
        )
        prompt = prompt_with_analyzer_context("Explain this result.", context)

        self.assertTrue(prompt.startswith("Explain this result."))
        self.assertIn('"panelId":"total_tps"', prompt)
        self.assertNotIn("exp.tp2.rate20.throughput", prompt)
        self.assertIn("Analyzer MCP", prompt)
        self.assertIn("do not invent tokens", prompt)


if __name__ == "__main__":
    unittest.main()
