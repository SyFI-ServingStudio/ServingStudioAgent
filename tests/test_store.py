import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.store import Store


class StoreMessagePageTest(unittest.TestCase):
    def test_pages_backwards_without_mutating_history(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            store_path = Path(temporary_directory) / "conversations.json"
            store = Store(store_path)
            store.create("conversation", "read-only")
            for message_index in range(7):
                store.add_message(
                    "conversation",
                    "user" if message_index % 2 == 0 else "assistant",
                    f"message-{message_index}",
                )

            newest_page = store.get_message_page(
                "conversation",
                before=None,
                limit=3,
            )
            assert newest_page is not None
            self.assertEqual(
                [message["content"] for message in newest_page["messages"]],
                ["message-4", "message-5", "message-6"],
            )
            self.assertEqual(
                newest_page["message_page"],
                {
                    "start_index": 4,
                    "end_index": 7,
                    "total_messages": 7,
                    "has_more": True,
                },
            )

            older_page = store.get_message_page(
                "conversation",
                before=newest_page["message_page"]["start_index"],
                limit=3,
            )
            assert older_page is not None
            self.assertEqual(
                [message["content"] for message in older_page["messages"]],
                ["message-1", "message-2", "message-3"],
            )
            self.assertTrue(older_page["message_page"]["has_more"])

            oldest_page = store.get_message_page(
                "conversation",
                before=older_page["message_page"]["start_index"],
                limit=3,
            )
            assert oldest_page is not None
            self.assertEqual(
                [message["content"] for message in oldest_page["messages"]],
                ["message-0"],
            )
            self.assertFalse(oldest_page["message_page"]["has_more"])

            newest_page["messages"][0]["content"] = "changed outside store"
            full_conversation = store.get("conversation")
            assert full_conversation is not None
            self.assertEqual(full_conversation["messages"][4]["content"], "message-4")


if __name__ == "__main__":
    unittest.main()
