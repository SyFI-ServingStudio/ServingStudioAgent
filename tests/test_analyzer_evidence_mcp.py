from __future__ import annotations

import json
import unittest
from email.message import Message
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.analyzer_evidence_mcp import server


class _JsonResponse(BytesIO):
    def __init__(self, payload: object) -> None:
        super().__init__(json.dumps(payload).encode())
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"

    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class AnalyzerEvidenceMcpTests(unittest.TestCase):
    def test_managed_kernel_curve_returns_metric_citation_map(self) -> None:
        resource_path = "/api/analyzer/v1/kernel-profiles/kp_test/subjects/curve/payload"
        curve_payload = {
            "series": [{"metric": "time_ms"}, {"metric": "tflops"}],
            "rows": [{"metrics": {"time_ms": 0.5, "tflops": 100.0}}],
        }
        citation_dictionary = {
            "entries": [
                {
                    "token": "kprof.curve.time_ms",
                    "target": {"kind": "kernel_profile", "metricKey": "time_ms"},
                },
                {
                    "token": "kprof.curve.tflops",
                    "target": {"kind": "kernel_profile", "metricKey": "tflops"},
                },
            ]
        }

        def open_request(request, timeout):
            if request.full_url.endswith(resource_path):
                return _JsonResponse(curve_payload)
            posted = json.loads(request.data)
            self.assertEqual(posted["resourceKind"], "kernel_profile")
            self.assertEqual(posted["profileId"], "kp_test")
            self.assertEqual(posted["analysis"], curve_payload)
            return _JsonResponse(citation_dictionary)

        with TemporaryDirectory() as temporary_directory:
            context_path = Path(temporary_directory) / "managed-run.json"
            context_path.write_text(
                json.dumps(
                    {
                        "backend_url": "http://backend.test:8765",
                        "capability_token": "token",
                    }
                ),
                "utf-8",
            )
            with (
                patch.dict(
                    "os.environ",
                    {"VIBESIM_MANAGED_RUN_CONTEXT": str(context_path)},
                ),
                patch.object(
                    server, "_base_url", return_value="http://analyzer.test:8787"
                ),
                patch.object(server, "urlopen", open_request),
            ):
                evidence = server.read_analyzer_resource(resource_path, source="host")

        self.assertEqual(evidence["result"], curve_payload)
        self.assertEqual(
            evidence["citations"],
            {
                "time_ms": "kprof.curve.time_ms",
                "tflops": "kprof.curve.tflops",
            },
        )

    def test_managed_prediction_returns_one_citation_adjacent_result(self) -> None:
        resource_path = (
            "/api/analyzer/v1/predictions/p_test/cases/40/"
            "subjects/optimality-waterfall/payload?mode=batch_locked"
        )
        prediction_payload = {
            "scope": "iteration",
            "rows": [{"name": "hardware_gap", "gpu_seconds": 1.25}],
        }
        citation_dictionary = {
            "identity": "prediction-test",
            "entries": [
                {
                    "token": "pred.casev40.batch_locked.optimality-breakdown",
                    "target": {
                        "kind": "prediction",
                        "predictionId": "p_test",
                        "caseId": "40",
                        "operationId": None,
                        "leafId": None,
                        "panelId": "optimality-breakdown",
                        "optimalityMode": "batch_locked",
                    },
                }
            ],
        }

        def open_request(request, timeout):
            if "/predictions/p_test/cases/40/subjects/optimality-waterfall" in request.full_url:
                return _JsonResponse(prediction_payload)
            posted = json.loads(request.data)
            self.assertEqual(
                posted,
                {
                    "resourceKind": "prediction",
                    "predictionId": "p_test",
                    "resourcePath": resource_path,
                },
            )
            return _JsonResponse(citation_dictionary)

        with TemporaryDirectory() as temporary_directory:
            context_path = Path(temporary_directory) / "managed-run.json"
            context_path.write_text(
                json.dumps(
                    {
                        "backend_url": "http://backend.test:8765",
                        "capability_token": "token",
                    }
                ),
                "utf-8",
            )
            with (
                patch.dict(
                    "os.environ",
                    {"VIBESIM_MANAGED_RUN_CONTEXT": str(context_path)},
                ),
                patch.object(
                    server, "_base_url", return_value="http://analyzer.test:8787"
                ),
                patch.object(server, "urlopen", open_request),
            ):
                evidence = server.read_analyzer_resource(resource_path, source="host")

        self.assertEqual(
            evidence,
            {
                "citation": "pred.casev40.batch_locked.optimality-breakdown",
                "result": prediction_payload,
            },
        )

    def test_managed_sweep_returns_compact_citation_adjacent_evidence(self) -> None:
        sweep_payload = {
            "protocol_version": 1,
            "schema_version": 1,
            "sweep_id": "e_test",
            "display_name": "20260801_0_llama_tp",
            "axes": ["tensor_parallel"],
            "metrics": [
                {
                    "key": "total_tps",
                    "label": "Total throughput",
                    "group": "throughput",
                    "unit": "tok/s",
                    "objective": "maximize",
                }
            ],
            "runs": [
                {
                    "run_id": "r_tp2",
                    "coordinates": {"tensor_parallel": 2},
                    "metrics": {"total_tps": 30124.2},
                }
            ],
        }
        citation_dictionary = {
            "identity": "aggregate-test",
            "document": "not returned to the Agent",
            "entries": [
                {
                    "token": "exp.throughput",
                    "target": {
                        "kind": "aggregate",
                        "experimentId": "e_test",
                        "metricKey": "total_tps",
                    },
                },
                {
                    "token": "exp.tp2.throughput",
                    "target": {
                        "kind": "aggregate",
                        "experimentId": "e_test",
                        "metricKey": "total_tps",
                        "runId": "r_tp2",
                    },
                },
            ],
        }

        def open_request(request, timeout):
            if request.full_url.endswith("/api/analyzer/v1/sweeps/e_test/subjects/sweep/payload"):
                return _JsonResponse(sweep_payload)
            self.assertTrue(
                request.full_url.endswith("/api/agent/v1/internal/analyzer-citations/register")
            )
            self.assertEqual(request.get_header("Authorization"), "Bearer token")
            posted = json.loads(request.data)
            self.assertEqual(posted["experimentId"], "e_test")
            self.assertEqual(posted["analysis"], sweep_payload)
            return _JsonResponse(citation_dictionary)

        with TemporaryDirectory() as temporary_directory:
            context_path = Path(temporary_directory) / "managed-run.json"
            context_path.write_text(
                json.dumps(
                    {
                        "backend_url": "http://backend.test:8765",
                        "capability_token": "token",
                    }
                ),
                "utf-8",
            )
            with (
                patch.dict(
                    "os.environ",
                    {"VIBESIM_MANAGED_RUN_CONTEXT": str(context_path)},
                ),
                patch.object(
                    server, "_base_url", return_value="http://analyzer.test:8787"
                ),
                patch.object(server, "urlopen", open_request),
            ):
                evidence = server.read_analyzer_resource(
                    "/api/analyzer/v1/sweeps/e_test/subjects/sweep/payload",
                    source="workspace",
                )

        self.assertEqual(evidence["resource"]["id"], "e_test")
        self.assertEqual(evidence["axes"], ["tensor_parallel"])
        self.assertEqual(
            evidence["metrics"]["total_tps"]["citation"],
            "exp.throughput",
        )
        self.assertEqual(
            evidence["rows"][0]["values"]["total_tps"],
            {"raw": 30124.2, "citation": "exp.tp2.throughput"},
        )
        self.assertNotIn("runs", evidence)
        self.assertNotIn("_vibesim_citations", evidence)

    def test_singleton_row_reuses_panel_citation(self) -> None:
        payload = {
            "protocol_version": 1,
            "schema_version": 1,
            "sweep_id": "e_singleton",
            "display_name": "glm52_ctx8k_c64",
            "axes": [],
            "metrics": [
                {
                    "key": "total_tps",
                    "label": "Total throughput",
                    "unit": "tok/s",
                    "objective": "maximize",
                }
            ],
            "runs": [
                {
                    "run_id": "r_singleton",
                    "coordinates": {},
                    "metrics": {"total_tps": 4110.69},
                }
            ],
        }
        dictionary = {
            "entries": [
                {
                    "token": "exp.throughput",
                    "target": {
                        "kind": "aggregate",
                        "experimentId": "e_singleton",
                        "metricKey": "total_tps",
                    },
                }
            ]
        }

        evidence = server._compact_sweep_evidence(payload, dictionary)

        self.assertEqual(
            evidence["metrics"]["total_tps"]["citation"],
            "exp.throughput",
        )
        self.assertEqual(
            evidence["rows"][0]["values"]["total_tps"],
            {"raw": 4110.69, "citation": "exp.throughput"},
        )

    def test_host_exact_sweep_uses_the_same_managed_registration_path(self) -> None:
        payload = {
            "protocol_version": 1,
            "schema_version": 1,
            "sweep_id": "e_host",
            "display_name": "host-sweep",
            "axes": [],
            "metrics": [],
            "runs": [],
        }
        dictionary = {"identity": "aggregate-host", "entries": []}

        def open_request(request, timeout):
            if request.full_url.endswith("/api/analyzer/v1/sweeps/e_host/subjects/sweep/payload"):
                return _JsonResponse(payload)
            self.assertTrue(
                request.full_url.endswith("/api/agent/v1/internal/analyzer-citations/register")
            )
            return _JsonResponse(dictionary)

        with TemporaryDirectory() as temporary_directory:
            context_path = Path(temporary_directory) / "managed-run.json"
            context_path.write_text(
                json.dumps(
                    {
                        "backend_url": "http://backend.test:8765",
                        "capability_token": "token",
                    }
                ),
                "utf-8",
            )
            with (
                patch.dict(
                    "os.environ",
                    {"VIBESIM_MANAGED_RUN_CONTEXT": str(context_path)},
                ),
                patch.object(
                    server, "_base_url", return_value="http://analyzer.test:8787"
                ),
                patch.object(server, "urlopen", open_request),
            ):
                evidence = server.read_analyzer_resource(
                    "/api/analyzer/v1/sweeps/e_host/subjects/sweep/payload",
                    source="host",
                )

        self.assertEqual(evidence["resource"], {"id": "e_host", "name": "host-sweep"})

    def test_read_resource_preserves_analyzer_json(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "ANALYZER_MCP_SOURCE": "external",
                    "ANALYZER_MCP_BASE_URL": "http://analyzer.test:8787",
                },
            ),
            patch.object(
                server,
                "urlopen",
                lambda request, timeout: _JsonResponse({"metric": 12.5}),
            ),
        ):
            self.assertEqual(
                server.read_analyzer_resource(
                    "/api/analyzer/v1/runs",
                    source="host",
                ),
                {"metric": 12.5},
            )

    def test_read_resource_rejects_paths_outside_protocol(self) -> None:
        for resource_path in (
            "https://example.com/api/analyzer/v1/runs",
            "/api/analyzer/v1/../secret",
            "/api/analyzer/v1/%2e%2e/secret",
            "/runs",
            "/api/analyzer/v1/runs#fragment",
        ):
            with self.subTest(resource_path=resource_path):
                with self.assertRaises(server.AnalyzerToolError):
                    server.validate_resource_path(resource_path)

    def test_compact_evidence_rejects_missing_or_duplicate_token_joins(self) -> None:
        payload = {
            "protocol_version": 1,
            "schema_version": 1,
            "sweep_id": "e_test",
            "display_name": "test",
            "axes": ["tensor_parallel"],
            "metrics": [
                {
                    "key": "total_tps",
                    "label": "Total throughput",
                    "unit": "tok/s",
                    "objective": "maximize",
                }
            ],
            "runs": [
                {
                    "run_id": "r_tp2",
                    "coordinates": {"tensor_parallel": 2},
                    "metrics": {"total_tps": 10.0},
                }
            ],
        }
        panel_entry = {
            "token": "exp.throughput",
            "target": {
                "kind": "aggregate",
                "experimentId": "e_test",
                "metricKey": "total_tps",
            },
        }

        with self.assertRaisesRegex(server.AnalyzerToolError, "0 matches.*r_tp2"):
            server._compact_sweep_evidence(
                payload,
                {"entries": [panel_entry]},
            )

        with self.assertRaisesRegex(server.AnalyzerToolError, "2 matches.*panel"):
            server._compact_sweep_evidence(
                payload,
                {"entries": [panel_entry, dict(panel_entry)]},
            )

    def test_exact_sweep_read_rejects_path_payload_identity_mismatch(self) -> None:
        with self.assertRaisesRegex(server.AnalyzerToolError, "identity"):
            server._exact_sweep_id(
                "/api/analyzer/v1/sweeps/e_path/subjects/sweep/payload",
                {"sweep_id": "e_payload"},
            )

    def test_compact_evidence_fails_instead_of_truncating(self) -> None:
        payload = {
            "protocol_version": 1,
            "schema_version": 1,
            "sweep_id": "e_test",
            "display_name": "test",
            "axes": [],
            "metrics": [],
            "runs": [],
        }
        with (
            patch.object(server, "MAX_EVIDENCE_BYTES", 1),
            self.assertRaisesRegex(server.AnalyzerToolError, "1 MiB"),
        ):
            server._compact_sweep_evidence(payload, {"entries": []})

    def test_mcp_exposes_one_generic_read_tool(self) -> None:
        self.assertEqual(server.TOOL_NAME, "read_analyzer_resource")
        self.assertIn("/api/analyzer/v1/sweeps?status=ready&limit=5", server.ENDPOINT_GUIDE)
        self.assertIn("/api/analyzer/v1/sweeps/latest", server.ENDPOINT_GUIDE)
        self.assertIn("adjacent complete citation token", server.ENDPOINT_GUIDE)

    def test_source_mode_is_explicit(self) -> None:
        with patch.dict("os.environ", {"ANALYZER_MCP_SOURCE": "unsupported"}):
            with self.assertRaisesRegex(
                server.AnalyzerToolError,
                "source must be 'host' or 'workspace'",
            ):
                server._base_url()
