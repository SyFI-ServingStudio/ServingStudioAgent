import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vibesim_agent.domain.conversations import RoleRuntime
from vibesim_agent.domain.roles import Role
from vibesim_agent.storage.conversations import Conversations
from vibesim_agent.storage.database import Database


class GeneratedTitleTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.database = Database.create(root / "database.sqlite")
        self.store = Conversations(self.database)
        runtime = RoleRuntime("provider", "scope", "model", "high", "default")
        self.store.create(
            "c", runtimes={role: runtime for role in Role}, naming_state="pending"
        )

    def test_pending_title_updates_once_and_missing_or_manual_returns_false(self):
        original = self.store.get("c")
        with patch("vibesim_agent.storage.conversations.time.time", return_value=100):
            self.assertTrue(self.store.apply_generated_title("c", "  Generated  "))
        current = self.store.get("c")
        self.assertEqual(
            current,
            original
            | {"title": "Generated", "naming_state": "generated", "updated_at": 100},
        )
        self.assertFalse(self.store.apply_generated_title("c", "Later"))
        self.assertFalse(self.store.apply_generated_title("missing", "Later"))
        with self.database.connect(write=True) as connection:
            connection.execute(
                "UPDATE conversations SET title = 'Manual', naming_state = 'manual' WHERE id = 'c'"
            )
        self.assertFalse(self.store.apply_generated_title("c", "Generated"))
        self.assertEqual(self.store.get("c")["title"], "Manual")

    def test_concurrent_generators_use_database_compare_and_set(self):
        def apply(index):
            return Conversations(self.database).apply_generated_title(
                "c", f"Name {index}"
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(apply, range(8)))
        self.assertEqual(sum(results), 1)
        self.assertEqual(self.store.get("c")["title"], f"Name {results.index(True)}")

    def test_empty_title_validation_does_not_mutate_pending_state(self):
        before = self.store.get("c")
        with self.assertRaises(ValueError):
            self.store.apply_generated_title("c", " \n")
        self.assertEqual(self.store.get("c"), before)
