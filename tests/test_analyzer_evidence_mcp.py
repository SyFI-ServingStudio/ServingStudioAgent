from __future__ import annotations

import json
import unittest
from email.message import Message
from io import BytesIO
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
                    "/api/v1/sweeps/s_1/payload",
                    source="host",
                ),
                {"metric": 12.5},
            )

    def test_read_resource_rejects_paths_outside_protocol(self) -> None:
        for resource_path in (
            "https://example.com/api/v1/runs",
            "/api/v1/../secret",
            "/api/v1/%2e%2e/secret",
            "/runs",
            "/api/v1/runs#fragment",
        ):
            with self.subTest(resource_path=resource_path):
                with self.assertRaises(server.AnalyzerToolError):
                    server.validate_resource_path(resource_path)

    def test_mcp_exposes_one_generic_read_tool(self) -> None:
        self.assertEqual(server.TOOL_NAME, "read_analyzer_resource")
        self.assertIn("/api/v1/sweeps", server.ENDPOINT_GUIDE)

    def test_source_mode_is_explicit(self) -> None:
        with patch.dict("os.environ", {"ANALYZER_MCP_SOURCE": "unsupported"}):
            with self.assertRaisesRegex(
                server.AnalyzerToolError,
                "source must be 'host' or 'workspace'",
            ):
                server._base_url()
