import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vibesim_agent.services import artifacts
from vibesim_agent.storage.registry import WorkspaceRegistry


class ArtifactServiceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.repo = self.root / "registry/w_test/repo"
        self.repo.mkdir(parents=True)
        self.logs = self.root / "external-logs"
        self.logs.mkdir()
        (self.repo.parent / "workspace.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "workspace_id": "w_test",
                    "display_name": "Test",
                    "state": "active",
                    "storage_kind": "managed",
                    "repo_path": "repo",
                    "logs_path": str(self.logs),
                    "created_at": 1,
                    "last_accessed_at": 1,
                }
            )
        )
        self.registry = WorkspaceRegistry(self.root / "registry")
        self.service = artifacts.ArtifactService(self.registry)
        self.golden = json.loads(
            (Path(__file__).parent / "fixtures/legacy_artifacts.json").read_text()
        )

    def write(self, name, value=b"text\n"):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        os.utime(path, (1, 1))
        return path

    def expected_listing(self, value):
        self.assertEqual(value["root"], "$REPO")
        value["root"] = str(self.repo)
        for entry in value["files"]:
            if entry.get("is_dir"):
                self.assertEqual(entry["size"], "$DIRECTORY_SIZE")
                entry["size"] = (self.repo / entry["path"]).stat().st_size
        return value

    def test_ordinary_metadata_listing_and_resolution_match_legacy(self):
        self.write("logs/summary.json", b'{"throughput": 12}\n')
        self.write("notes.md", b"# Results\n")
        self.write("plot.png", b"image")
        self.write("target/ignored", b"hidden")
        for path in self.repo.rglob("*"):
            os.utime(path, (1, 1))
        golden = self.golden["ordinary"]
        for recursive, preview, key in (
            (True, False, "recursive"),
            (False, True, "shallow"),
        ):
            self.assertEqual(
                self.service.list_artifacts(
                    "w_test", recursive=recursive, preview=preview
                ),
                self.expected_listing(golden[key]),
            )
        self.assertEqual(
            self.service.artifact_meta("w_test", "notes.md"), golden["meta"]
        )
        self.assertEqual(
            self.service.resolve_preview("w_test", "/workspace/notes.md"),
            (self.repo / golden["preview"][0], golden["preview"][1]),
        )
        self.assertEqual(
            self.service.resolve_artifact("w_test", "target/ignored"),
            self.repo / "target/ignored",
        )
        with self.assertRaises(PermissionError):
            self.service.resolve_preview("w_test", "target/ignored")

    def test_declared_external_logs_remain_owned_and_keep_old_relative_listing_shape(
        self,
    ):
        path = self.logs / "summary.txt"
        path.write_text("external result")
        tree = self.service.workspace_tree("w_test")
        self.assertEqual(tree.roots, (self.repo, self.logs))
        self.assertEqual(
            self.service.resolve_preview("w_test", str(path)), (path, "text")
        )
        listing = self.service.list_artifacts("w_test", subdir=str(self.logs))
        self.assertEqual(listing["path"], ".")
        self.assertEqual(listing["files"][0]["path"], "summary.txt")
        self.assertEqual(listing["root"], str(self.repo))

    def test_absolute_host_alias_into_owned_repo_preserves_legacy_preview(self):
        target = self.write("result.txt")
        alias = self.root / "alias-to-repo"
        alias.symlink_to(self.repo, target_is_directory=True)
        requested = str(alias / "result.txt")
        golden = self.golden["alias"]
        self.assertEqual(
            self.service.resolve_preview("w_test", requested),
            (self.repo / golden["preview"][0], golden["preview"][1]),
        )
        self.assertEqual(
            self.service.artifact_meta("w_test", requested), golden["meta"]
        )
        self.assertEqual(
            self.service.list_artifacts("w_test", subdir=str(alias), preview=True),
            self.expected_listing(golden["listing"]),
        )
        self.assertEqual(
            self.service.resolve_preview("w_test", requested), (target, "text")
        )
        credential_alias = self.root / "credentials"
        credential_alias.symlink_to(target)
        with self.assertRaises(PermissionError):
            self.service.resolve_preview("w_test", str(credential_alias))
        self.write(".env", b"private")
        with self.assertRaises(PermissionError):
            self.service.resolve_preview("w_test", str(alias / ".env"))

    def test_each_listing_entry_is_guarded_before_metadata_and_keeps_safe_alias_name(
        self,
    ):
        safe = self.write("safe.txt")
        secret = self.write(".env", b"private")
        outside = self.root / "outside"
        outside.write_bytes(b"outside")
        (self.repo / "alias.txt").symlink_to(safe)
        (self.repo / "credential-alias.txt").symlink_to(secret)
        (self.repo / "outside-alias.txt").symlink_to(outside)
        (self.repo / "broken").symlink_to(self.root / "missing")
        (self.repo / ".env.safealias").symlink_to(safe)
        for recursive in (True, False):
            with self.subTest(recursive=recursive):
                listed = self.service.list_artifacts(
                    "w_test", recursive=recursive, preview=True
                )
                self.assertEqual(
                    {item["path"] for item in listed["files"]},
                    {"alias.txt", "safe.txt"},
                )
                privileged = self.service.list_artifacts("w_test", recursive=recursive)
                self.assertNotIn(
                    "outside-alias.txt", {item["path"] for item in privileged["files"]}
                )
                self.assertIn(
                    "credential-alias.txt",
                    {item["path"] for item in privileged["files"]},
                )
        for name in ("credential-alias.txt", ".env.safealias"):
            with self.assertRaises(PermissionError):
                self.service.resolve_preview("w_test", name)
        with self.assertRaises(PermissionError):
            self.service.resolve_artifact("w_test", "outside-alias.txt")

    def test_directory_symlinks_are_not_recursively_followed_and_preview_aliases_are_denied(
        self,
    ):
        self.write("data/file.txt")
        self.write(".git/config")
        (self.repo / "data-alias").symlink_to(
            self.repo / "data", target_is_directory=True
        )
        (self.repo / "vcs-alias").symlink_to(
            self.repo / ".git", target_is_directory=True
        )
        outside = self.root / "outside-dir"
        outside.mkdir()
        (outside / "secret").write_text("hidden")
        (self.repo / "outside-dir").symlink_to(outside, target_is_directory=True)
        shallow = self.service.list_artifacts("w_test", recursive=False, preview=True)
        self.assertEqual(
            {entry["path"] for entry in shallow["files"]}, {"data", "data-alias"}
        )
        deep = self.service.list_artifacts("w_test", preview=True)
        self.assertEqual([entry["path"] for entry in deep["files"]], ["data/file.txt"])

    def test_disappearing_entry_is_skipped_without_losing_other_files(self):
        self.write("gone")
        self.write("keep")
        original = artifacts.guard_preview

        def guard(tree, path):
            if Path(path).name == "gone":
                raise FileNotFoundError("entry disappeared")
            return original(tree, path)

        for recursive in (True, False):
            with patch.object(artifacts, "guard_preview", side_effect=guard):
                listing = self.service.list_artifacts(
                    "w_test", recursive=recursive, preview=True
                )
            self.assertEqual([entry["path"] for entry in listing["files"]], ["keep"])

    def test_limits_and_guard_failures_preserve_contract(self):
        self.write("a.txt")
        self.write("b.txt")
        listing = self.service.list_artifacts("w_test", limit=1)
        self.assertEqual(listing["count"], 1)
        self.assertTrue(listing["truncated"])
        with self.assertRaises(FileNotFoundError):
            self.service.workspace_tree("w_missing")
        with self.assertRaises(FileNotFoundError):
            self.service.resolve_artifact("w_test", "missing")
        with self.assertRaises(FileNotFoundError):
            self.service.resolve_preview("w_test", ".")
        outside = self.root / "outside"
        outside.write_text("outside")
        with self.assertRaises(PermissionError):
            self.service.artifact_meta("w_test", str(outside))

    def test_utf8_sniff_accepts_split_valid_character_and_rejects_invalid_tail(self):
        valid = self.write("valid", b"a" * 8186 + "\u20ac\U0001f600".encode())
        invalid = self.write("invalid", b"a" * 8191 + b"\xff")
        incomplete = self.write("incomplete", b"abc\xc3")
        for path in (valid, invalid, incomplete):
            self.assertNotEqual(
                artifacts.preview_kind(path), self.golden["legacy_sniff"][path.name]
            )
        self.assertEqual(artifacts.preview_kind(valid), "text")
        self.assertEqual(artifacts.preview_kind(invalid), "binary")
        self.assertEqual(artifacts.preview_kind(incomplete), "binary")

    def test_bounded_preview_and_classification_match_existing_vectors(self):
        path = self.write("report.txt", b"first\nsecond\nthird\nfourth\n")
        with patch.object(artifacts, "MAX_PREVIEW_BYTES", 16):
            self.assertEqual(
                artifacts.read_text_preview(path), tuple(self.golden["bounded_preview"])
            )
            self.assertEqual(
                artifacts.read_text_preview(path), ("first\nsecond\n", True)
            )
        with patch.object(artifacts, "MAX_PREVIEW_LINES", 2):
            self.assertEqual(artifacts.read_text_preview(path), ("first\nsecond", True))
        for name, raw, kind, language in (
            ("Justfile", b"all:\n", "text", "makefile"),
            ("x.py", b"print(1)", "text", "python"),
            ("blob", b"a\0b", "binary", None),
            ("x.parquet", b"text", "binary", None),
            ("x.png", b"image", "image", None),
        ):
            path = self.write(name, raw)
            self.assertEqual(artifacts.preview_kind(path), kind)
            self.assertEqual(artifacts.language_for(path), language)
