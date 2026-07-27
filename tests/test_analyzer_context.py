from __future__ import annotations

import unittest
from pydantic import ValidationError

from backend.analyzer_context import (
    AnalyzerTurnContext,
    CitationDictionarySnapshot,
    freeze_citations,
    prompt_with_analyzer_context,
)


def dictionary() -> CitationDictionarySnapshot:
    return CitationDictionarySnapshot.model_validate(
        {
            "protocol": "vibesim.citation-dictionary/v1",
            "identity": "s_test:1",
            "document": "Use `exp.tp2.rate20.throughput`.",
            "entries": [
                {
                    "token": "exp.tp2.rate20.throughput",
                    "displayLabel": "TP=2 · rate=20 · Total throughput",
                    "target": {
                        "protocol": "vibesim.analyzer/v1",
                        "kind": "aggregate",
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
            "protocol": "vibesim.analyzer/v1",
            "kind": "run",
            "runId": "r_test",
            "scope": "worker",
        }
        with self.assertRaises(ValidationError):
            CitationDictionarySnapshot.model_validate(payload)

    def test_prompt_exposes_literal_selection_and_bounded_dictionary(self) -> None:
        context = AnalyzerTurnContext.model_validate(
            {
                "protocol": "vibesim.conversation-context/v1",
                "selection": {
                    "kind": "aggregate",
                    "experimentId": "s_test",
                    "panelId": "total_tps",
                },
                "citationDictionary": dictionary().model_dump(by_alias=True),
            }
        )
        prompt = prompt_with_analyzer_context("Explain this result.", context)

        self.assertTrue(prompt.startswith("Explain this result."))
        self.assertIn('"panelId":"total_tps"', prompt)
        self.assertIn("exp.tp2.rate20.throughput", prompt)
        self.assertIn("Do not invent tokens", prompt)


if __name__ == "__main__":
    unittest.main()
