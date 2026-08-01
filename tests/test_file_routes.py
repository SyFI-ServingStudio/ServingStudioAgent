"""Untokened workspace file preview: /api/file, /api/file/meta, /api/file/list.

These routes widened from an image-only whitelist to any workspace file, so the
guard is the contract: containment inside the workspace, a build/VCS/credential
denylist, and a bounded read. Every test below pins one of those.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator
from unittest.mock import patch

from fastapi import HTTPException

from backend import app as app_module
from backend import artifacts as artifacts_module
from backend.store import WorkspaceRegistry

WORKSPACE_ID = "w_preview"


@contextmanager
def workspace() -> Iterator[Path]:
    """A registered workspace whose repo is empty and ready to be filled."""
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        main_dir = root / "main"
        (main_dir / "logs").mkdir(parents=True)
        registry = WorkspaceRegistry(root / "agent-workspaces", main_dir=main_dir)
        registry.create("Preview", workspace_id=WORKSPACE_ID)
        repo = registry.repo_path(WORKSPACE_ID)
        (repo / "logs").mkdir(parents=True, exist_ok=True)
        # artifacts.py builds its own registry per call; point it at the temp one.
        with patch.object(artifacts_module, "WorkspaceRegistry", lambda: registry):
            yield repo


def status_of(call) -> int:
    try:
        call()
    except HTTPException as exc:
        return exc.status_code
    return 200


class FilePreviewRouteTests(unittest.TestCase):
    def test_serves_a_text_file_as_utf8_plain_text(self) -> None:
        with workspace() as repo:
            (repo / "logs" / "summary.json").write_text('{"total_tps": 4195}\n')

            response = app_module.serve_file(
                path="logs/summary.json", workspace_id=WORKSPACE_ID
            )

            self.assertEqual(response.media_type, "text/plain; charset=utf-8")
            self.assertEqual(response.body.decode(), '{"total_tps": 4195}\n')
            self.assertEqual(response.headers["x-file-truncated"], "0")

    def test_keeps_serving_images_so_markdown_plots_do_not_regress(self) -> None:
        with workspace() as repo:
            plot = repo / "logs" / "throughput.png"
            plot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

            response = app_module.serve_file(
                path="logs/throughput.png", workspace_id=WORKSPACE_ID
            )

            self.assertEqual(Path(response.path), plot)

    def test_binary_is_offered_as_a_download_not_as_text(self) -> None:
        with workspace() as repo:
            (repo / "logs" / "worker_cost.parquet").write_bytes(b"PAR1\x00\x01")

            response = app_module.serve_file(
                path="logs/worker_cost.parquet", workspace_id=WORKSPACE_ID
            )

            self.assertEqual(response.media_type, "application/octet-stream")
            self.assertIn("attachment", response.headers["content-disposition"])

    def test_extensionless_text_still_previews_as_text(self) -> None:
        with workspace() as repo:
            (repo / "Justfile").write_text("test:\n    cargo test\n")

            meta = app_module.serve_file_meta(path="Justfile", workspace_id=WORKSPACE_ID)

            self.assertEqual(meta["preview_kind"], "text")
            self.assertEqual(meta["language"], "makefile")

    def test_container_absolute_paths_resolve_against_the_repo(self) -> None:
        with workspace() as repo:
            (repo / "logs" / "run.log").write_text("started\n")

            response = app_module.serve_file(
                path="/workspace/logs/run.log", workspace_id=WORKSPACE_ID
            )

            self.assertEqual(response.body.decode(), "started\n")

    def test_a_path_escaping_the_workspace_is_refused(self) -> None:
        with workspace() as repo:
            (repo.parent.parent / "outside.txt").write_text("secret\n")

            self.assertEqual(
                status_of(
                    lambda: app_module.serve_file(
                        path="../../outside.txt", workspace_id=WORKSPACE_ID
                    )
                ),
                403,
            )

    def test_vcs_internals_are_refused(self) -> None:
        with workspace() as repo:
            (repo / ".git").mkdir()
            (repo / ".git" / "config").write_text("[remote]\n")

            self.assertEqual(
                status_of(
                    lambda: app_module.serve_file(
                        path=".git/config", workspace_id=WORKSPACE_ID
                    )
                ),
                403,
            )

    def test_credential_shaped_files_are_refused(self) -> None:
        with workspace() as repo:
            (repo / ".env").write_text("TOKEN=abc\n")
            (repo / "deploy.pem").write_text("-----BEGIN KEY-----\n")

            for path in (".env", "deploy.pem"):
                with self.subTest(path=path):
                    self.assertEqual(
                        status_of(
                            lambda: app_module.serve_file(
                                path=path, workspace_id=WORKSPACE_ID
                            )
                        ),
                        403,
                    )

    def test_oversized_text_is_truncated_on_a_line_boundary(self) -> None:
        with workspace() as repo:
            (repo / "big.log").write_text("".join(f"line {index}\n" for index in range(50)))

            with patch.object(artifacts_module, "MAX_PREVIEW_BYTES", 40):
                response = app_module.serve_file(
                    path="big.log", workspace_id=WORKSPACE_ID
                )

            body = response.body.decode()
            self.assertEqual(response.headers["x-file-truncated"], "1")
            self.assertTrue(body.endswith("\n"))
            self.assertLessEqual(len(body), 40)
            self.assertGreater(int(response.headers["x-file-total-bytes"]), 40)

    def test_meta_describes_a_directory_without_reading_it(self) -> None:
        with workspace() as repo:
            (repo / "logs" / "run" ).mkdir()

            meta = app_module.serve_file_meta(path="logs/run", workspace_id=WORKSPACE_ID)

            self.assertTrue(meta["is_dir"])
            self.assertEqual(meta["preview_kind"], "directory")
            self.assertEqual(meta["path"], "logs/run")

    def test_listing_shows_one_level_and_hides_denied_entries(self) -> None:
        with workspace() as repo:
            (repo / "logs" / "nested").mkdir()
            (repo / "logs" / "nested" / "deep.txt").write_text("deep\n")
            (repo / "logs" / "summary.json").write_text("{}\n")
            (repo / "logs" / ".env").write_text("TOKEN=abc\n")

            # Called directly, so FastAPI's Query default is not resolved for us.
            listing = app_module.serve_file_list(
                path="logs", workspace_id=WORKSPACE_ID, limit=2000
            )

            names = [entry["name"] for entry in listing["files"]]
            self.assertEqual(names, ["nested", "summary.json"])
            self.assertTrue(listing["files"][0]["is_dir"])

    def test_an_unknown_workspace_is_not_found(self) -> None:
        with workspace():
            self.assertEqual(
                status_of(
                    lambda: app_module.serve_file(
                        path="logs/summary.json", workspace_id="w_missing"
                    )
                ),
                404,
            )

    def test_a_missing_file_is_not_found(self) -> None:
        with workspace():
            self.assertEqual(
                status_of(
                    lambda: app_module.serve_file(
                        path="logs/absent.json", workspace_id=WORKSPACE_ID
                    )
                ),
                404,
            )


if __name__ == "__main__":
    unittest.main()
