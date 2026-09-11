import base64
import json
import unittest
from unittest.mock import patch

import httpx
from pydantic import SecretStr
from tests import test_application as fixtures
from vibesim_agent.storage.database import Database


class FileHttpTests(unittest.IsolatedAsyncioTestCase):
    app = fixtures.ApplicationTests.app
    providers = fixtures.ApplicationTests.providers
    base = "/api/agent/v1/file"
    tools = "/api/agent/v1/tools/workspaces/w_main/artifacts"

    def setUp(self):
        fixtures.ApplicationTests.setUp(self)
        self.headers = {"Authorization": "Bearer token"}
        self.settings = self.settings.model_copy(
            update={
                "agent": self.settings.agent.model_copy(
                    update={"api_token": SecretStr("token")}
                )
            }
        )
        (self.repo / "logs").mkdir()
        (self.repo / "logs/result.rs").write_text("one\ntwo\nthree\n")
        (self.repo / "logs/blob.dat").write_bytes(b"binary\0content")
        self.png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="
        )
        (self.repo / "logs/plot.png").write_bytes(self.png)

    async def asyncSetUp(self):
        Database.create(self.state / "workspace.sqlite")
        self.application = self.app()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.application), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.addAsyncCleanup(self.application.state.evaluations.close)

    async def test_public_text_preview_metadata_and_bounded_headers(self):
        before = sorted(path.name for path in self.state.iterdir())
        response = await self.client.get(
            self.base, params={"path": "/workspace/logs/result.rs"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.text, "one\ntwo\nthree\n")
        self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")
        self.assertEqual(response.headers["x-file-truncated"], "0")
        self.assertEqual(response.headers["x-file-total-bytes"], "14")
        metadata = (
            await self.client.get(
                self.base + "/meta", params={"path": "logs/result.rs"}
            )
        ).json()
        self.assertEqual(metadata["path"], "logs/result.rs")
        self.assertEqual(metadata["workspace_id"], "w_main")
        self.assertEqual(metadata["preview_kind"], "text")
        self.assertEqual(metadata["language"], "rust")
        self.assertFalse(metadata["is_dir"])
        self.assertEqual(metadata["preview_byte_limit"], 5 * 1024 * 1024)
        with patch("vibesim_agent.services.artifacts.MAX_PREVIEW_BYTES", 8):
            clipped = await self.client.get(
                self.base, params={"path": "logs/result.rs"}
            )
        self.assertEqual(clipped.text, "one\ntwo\n")
        self.assertEqual(clipped.headers["x-file-truncated"], "1")
        self.assertEqual(clipped.headers["x-file-total-bytes"], "14")
        self.assertEqual(sorted(path.name for path in self.state.iterdir()), before)
        self.assertEqual(self.docker.calls, [])

    async def test_image_inline_binary_attachment_and_tools_download(self):
        image = await self.client.get(self.base, params={"path": "logs/plot.png"})
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.content, self.png)
        self.assertEqual(image.headers["content-type"], "image/png")
        self.assertNotIn("content-disposition", image.headers)
        binary = await self.client.get(self.base, params={"path": "logs/blob.dat"})
        self.assertEqual(binary.content, b"binary\0content")
        self.assertEqual(binary.headers["content-type"], "application/octet-stream")
        self.assertIn(
            'attachment; filename="blob.dat"', binary.headers["content-disposition"]
        )
        downloaded = await self.client.get(
            self.tools + "/download",
            headers=self.headers,
            params={"path": "logs/plot.png"},
        )
        self.assertEqual(downloaded.content, self.png)
        self.assertIn("attachment", downloaded.headers["content-disposition"])

    async def test_browser_denies_credentials_and_aliases_tools_requires_token(self):
        (self.repo / ".env").write_text("private")
        (self.repo / "alias.txt").symlink_to(".env")
        (self.repo / ".git").mkdir()
        (self.repo / ".git/config").write_text("vcs")
        for path in (".env", "alias.txt", ".git/config"):
            for endpoint in (self.base, self.base + "/meta"):
                response = await self.client.get(endpoint, params={"path": path})
                self.assertEqual(response.status_code, 403, response.text)
        for endpoint in (self.tools, self.tools + "/download"):
            self.assertEqual(
                (await self.client.get(endpoint, params={"path": ".env"})).status_code,
                401,
            )
        download = await self.client.get(
            self.tools + "/download", headers=self.headers, params={"path": ".env"}
        )
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.text, "private")
        listed = (await self.client.get(self.base + "/list")).json()["files"]
        self.assertFalse(
            {".env", "alias.txt", ".git"} & {item["name"] for item in listed}
        )

    async def test_list_shapes_recursion_limits_and_outside_links(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside")
        (self.repo / "logs/escape.txt").symlink_to(outside)
        nested = self.repo / "logs/nested"
        nested.mkdir()
        (nested / "child.txt").write_text("child")
        browser = (
            await self.client.get(self.base + "/list", params={"path": "logs"})
        ).json()
        self.assertEqual(browser["root"], str(self.repo))
        self.assertEqual(browser["path"], "logs")
        self.assertEqual(browser["count"], 4)
        self.assertEqual(browser["files"][0]["name"], "nested")
        self.assertTrue(browser["files"][0]["is_dir"])
        self.assertFalse(browser["truncated"])
        recursive = (
            await self.client.get(
                self.tools, headers=self.headers, params={"subdir": "logs"}
            )
        ).json()
        self.assertEqual(recursive["count"], 4)
        self.assertIn(
            "logs/nested/child.txt", [item["path"] for item in recursive["files"]]
        )
        self.assertNotIn(
            "logs/escape.txt", [item["path"] for item in recursive["files"]]
        )
        self.assertTrue(
            all(set(item) == {"path", "size", "mtime"} for item in recursive["files"])
        )
        limited = (
            await self.client.get(
                self.tools, headers=self.headers, params={"subdir": "logs", "limit": 1}
            )
        ).json()
        self.assertEqual(limited["count"], 1)
        self.assertTrue(limited["truncated"])
        for endpoint, headers in (
            (self.base + "/list", {}),
            (self.tools, self.headers),
        ):
            for limit in (0, 20001):
                self.assertEqual(
                    (
                        await self.client.get(
                            endpoint, headers=headers, params={"limit": limit}
                        )
                    ).status_code,
                    422,
                )
        escaped = await self.client.get(self.base, params={"path": "logs/escape.txt"})
        self.assertEqual(escaped.status_code, 403)

    async def test_external_logs_root_and_missing_path_errors(self):
        logs = self.root / "external-logs"
        logs.mkdir()
        report = logs / "result.json"
        report.write_text('{"result": 1}')
        descriptor_path = self.state / "workspace.json"
        descriptor = json.loads(descriptor_path.read_text())
        descriptor["logs_path"] = str(logs)
        descriptor_path.write_text(json.dumps(descriptor))
        preview = await self.client.get(self.base, params={"path": str(report)})
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.json(), {"result": 1})
        metadata = (
            await self.client.get(self.base + "/meta", params={"path": str(report)})
        ).json()
        self.assertEqual(metadata["path"], "result.json")
        listed = (
            await self.client.get(self.base + "/list", params={"path": str(logs)})
        ).json()
        self.assertEqual(listed["path"], ".")
        self.assertEqual(listed["root"], str(self.repo))
        self.assertEqual(listed["files"][0]["path"], "result.json")
        for params in (
            {"path": "missing"},
            {"path": "logs"},
            {"path": "AGENTS.md", "workspace_id": "w_missing"},
        ):
            response = await self.client.get(self.base, params=params)
            self.assertEqual(response.status_code, 404, response.text)
        for endpoint in (self.base, self.base + "/meta"):
            self.assertEqual((await self.client.get(endpoint)).status_code, 422)
