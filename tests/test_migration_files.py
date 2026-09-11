import hashlib
import os
import shutil
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools import migration_files


class MigrationFilesTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.source = self.root / "source"
        self.source.mkdir()
        self.target = self.root / "target"
        self.outside = self.root / "outside"
        self.outside.mkdir()
        (self.outside / "private").write_bytes(b"outside bytes")
        (self.source / "nested").mkdir()
        (self.source / ".hidden").write_bytes(b"dirty\x00\xff")
        (self.source / "nested/run").write_bytes(b"#!/bin/sh\nexit 0\n")
        os.link(self.source / ".hidden", self.source / "nested/hardlink")
        (self.source / "relative").symlink_to("nested/../.hidden")
        (self.source / "dangling").symlink_to("missing/../nowhere")
        (self.source / "external").symlink_to(self.outside, target_is_directory=True)
        for index, path in enumerate(sorted(self.source.rglob("*"))):
            if not path.is_symlink():
                path.chmod(0o750 if path.is_dir() or path.name == "run" else 0o640)
            os.utime(
                path,
                ns=(
                    1_650_000_000_123_000_000 + index,
                    1_650_000_000_456_000_000 + index,
                ),
                follow_symlinks=False,
            )
        self.source.chmod(0o751)
        os.utime(self.source, ns=(1_650_000_000_000_000_001, 1_650_000_000_000_000_002))

    def snapshot(self, root):
        paths = [root, *sorted(root.rglob("*"))]
        result = {}
        for path in paths:
            info = path.lstat()
            payload = (
                os.readlink(path)
                if path.is_symlink()
                else path.read_bytes()
                if path.is_file()
                else None
            )
            result[str(path.relative_to(root))] = (
                stat.S_IMODE(info.st_mode),
                info.st_mtime_ns,
                payload,
            )
        return result

    def test_inventory_has_sorted_exact_metadata_and_hardlink_byte_accounting(self):
        inventory = migration_files.inventory_tree(self.source)
        entries = inventory["entries"]
        self.assertEqual(
            [entry["path"] for entry in entries], sorted(self.snapshot(self.source))
        )
        by_path = {entry["path"]: entry for entry in entries}
        self.assertEqual(by_path["."]["kind"], "directory")
        for relative, (mode, mtime, _) in self.snapshot(self.source).items():
            self.assertEqual(by_path[relative]["mode"], mode)
            self.assertEqual(by_path[relative]["mtime_ns"], mtime)
        self.assertEqual(by_path[".hidden"]["hardlink_to"], None)
        self.assertEqual(by_path["nested/hardlink"]["hardlink_to"], ".hidden")
        self.assertEqual(
            by_path[".hidden"]["sha256"], hashlib.sha256(b"dirty\x00\xff").hexdigest()
        )
        self.assertEqual(by_path[".hidden"]["size"], 7)
        self.assertEqual(by_path["relative"]["target"], "nested/../.hidden")
        self.assertEqual(by_path["dangling"]["target"], "missing/../nowhere")
        self.assertEqual(by_path["external"]["target"], str(self.outside))
        self.assertNotIn("external/private", by_path)
        self.assertEqual(
            inventory["logical_bytes"], 7 * 2 + len(b"#!/bin/sh\nexit 0\n")
        )
        self.assertEqual(inventory["unique_bytes"], 7 + len(b"#!/bin/sh\nexit 0\n"))

    def test_copy_preserves_raw_links_modes_times_and_only_internal_target_hardlinks(
        self,
    ):
        before = self.snapshot(self.source)
        expected = migration_files.inventory_tree(self.source)
        copied = migration_files.copy_tree(self.source, self.target)
        self.assertEqual(copied, expected)
        self.assertEqual(self.snapshot(self.target), before)
        self.assertEqual(self.snapshot(self.source), before)
        source_inode = (self.source / ".hidden").stat().st_ino
        destination_inode = (self.target / ".hidden").stat().st_ino
        self.assertNotEqual(source_inode, destination_inode)
        self.assertEqual(
            destination_inode, (self.target / "nested/hardlink").stat().st_ino
        )
        (self.target / ".hidden").write_bytes(b"target edit")
        self.assertEqual((self.target / "nested/hardlink").read_bytes(), b"target edit")
        self.assertEqual((self.source / ".hidden").read_bytes(), b"dirty\x00\xff")
        self.assertEqual((self.outside / "private").read_bytes(), b"outside bytes")

    def test_existing_and_dangling_target_are_never_overwritten(self):
        self.target.mkdir()
        with self.assertRaises(FileExistsError):
            migration_files.copy_tree(self.source, self.target)
        self.assertEqual(list(self.target.iterdir()), [])
        self.target.rmdir()
        missing = self.root / "missing-target"
        self.target.symlink_to(missing, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            migration_files.copy_tree(self.source, self.target)
        self.assertEqual(os.readlink(self.target), str(missing))
        self.assertFalse(missing.exists())

    def test_source_overlap_and_alias_overlap_are_rejected_without_changes(self):
        before = self.snapshot(self.source)
        alias = self.root / "source-alias"
        alias.symlink_to(self.source, target_is_directory=True)
        for target in (self.source, self.source / "child", self.root, alias / "child"):
            with (
                self.subTest(target=target),
                self.assertRaises((ValueError, FileExistsError)),
            ):
                migration_files.copy_tree(self.source, target)
        self.assertEqual(self.snapshot(self.source), before)

    def test_fifo_is_rejected_without_copy_or_opening_it(self):
        fifo = self.source / "fifo"
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            migration_files.inventory_tree(self.source)
        with self.assertRaises(ValueError):
            migration_files.copy_tree(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertTrue(stat.S_ISFIFO(fifo.lstat().st_mode))

    def test_source_change_after_inventory_fails_and_cleans_owned_target(self):
        scan = migration_files._scan
        changed = False

        def capture(*args, **kwargs):
            nonlocal changed
            result = scan(*args, **kwargs)
            if not changed:
                changed = True
                (self.source / "nested/run").write_bytes(b"changed source bytes")
            return result

        with (
            patch.object(migration_files, "_scan", side_effect=capture),
            self.assertRaises((ValueError, RuntimeError)),
        ):
            migration_files.copy_tree(self.source, self.target)
        self.assertTrue(changed)
        self.assertFalse(self.target.exists())
        self.assertEqual(
            (self.source / "nested/run").read_bytes(), b"changed source bytes"
        )

    def test_source_root_symlink_is_rejected_without_following_it(self):
        alias = self.root / "source-link"
        alias.symlink_to(self.source, target_is_directory=True)
        before = self.snapshot(self.source)
        with self.assertRaises((ValueError, OSError)):
            migration_files.inventory_tree(alias)
        with self.assertRaises((ValueError, OSError)):
            migration_files.copy_tree(alias, self.target)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.snapshot(self.source), before)

    def test_directory_replaced_by_external_symlink_during_copy_is_rejected(self):
        (self.outside / "run").write_bytes(b"outside run")
        (self.outside / "hardlink").write_bytes(b"outside hardlink")
        scan = migration_files._scan
        changed = False

        def replace_directory(*args, **kwargs):
            nonlocal changed
            result = scan(*args, **kwargs)
            if not changed:
                changed = True
                (self.source / "nested").rename(self.source / "original-nested")
                (self.source / "nested").symlink_to(
                    self.outside, target_is_directory=True
                )
            return result

        with (
            patch.object(migration_files, "_scan", side_effect=replace_directory),
            self.assertRaises((ValueError, RuntimeError, OSError)),
        ):
            migration_files.copy_tree(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertEqual((self.outside / "private").read_bytes(), b"outside bytes")
        self.assertTrue((self.source / "original-nested/run").exists())

    def test_partial_copy_failure_cleans_owned_target_without_source_changes(self):
        before = self.snapshot(self.source)

        def fail_copy(source_fd, target_fd):
            os.write(target_fd, b"partial bytes")
            raise OSError("injected copy failure")

        with (
            patch.object(migration_files, "_copy_file", side_effect=fail_copy),
            self.assertRaisesRegex(OSError, "injected copy failure"),
        ):
            migration_files.copy_tree(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.snapshot(self.source), before)

    def test_new_source_entry_after_last_file_copy_is_detected_by_final_rescan(self):
        copy_file = migration_files._copy_file
        copies = 0

        def add_after_copy(source_fd, target_fd):
            nonlocal copies
            copy_file(source_fd, target_fd)
            copies += 1
            if copies == 2:
                (self.source / "late-entry").write_bytes(b"new source content")

        with (
            patch.object(migration_files, "_copy_file", side_effect=add_after_copy),
            self.assertRaises(ValueError),
        ):
            migration_files.copy_tree(self.source, self.target)
        self.assertEqual(copies, 2)
        self.assertEqual(
            (self.source / "late-entry").read_bytes(), b"new source content"
        )
        self.assertFalse(self.target.exists())

    def test_failure_does_not_delete_foreign_replacement_of_owned_target(self):
        displaced = self.root / "owned-moved"

        def replace_target(source_fd, target_fd):
            os.write(target_fd, b"partial copy")
            self.target.rename(displaced)
            self.target.mkdir()
            (self.target / "foreign").write_bytes(b"must stay")
            raise OSError("injected replacement")

        try:
            with (
                patch.object(migration_files, "_copy_file", side_effect=replace_target),
                self.assertRaisesRegex(OSError, "injected replacement"),
            ):
                migration_files.copy_tree(self.source, self.target)
            self.assertEqual((self.target / "foreign").read_bytes(), b"must stay")
            self.assertTrue(displaced.exists())
        finally:
            if displaced.exists():
                shutil.rmtree(displaced)

    def test_same_size_corrupted_copy_is_rejected_by_target_digest(self):
        before = self.snapshot(self.source)

        def corrupt_copy(source_fd, target_fd):
            remaining = os.fstat(source_fd).st_size
            while remaining:
                remaining -= os.write(target_fd, b"x" * remaining)

        with (
            patch.object(migration_files, "_copy_file", side_effect=corrupt_copy),
            self.assertRaises(ValueError),
        ):
            migration_files.copy_tree(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.snapshot(self.source), before)
