import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory import conversation_memory


class ConversationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_conversations.sqlite3"
        self.memory_patch = patch.object(conversation_memory, "MEMORY_FILE", self.db_path)
        self.memory_patch.start()
        self.addCleanup(self.memory_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def test_save_and_load_messages_sqlite(self):
        conversation_memory.save_message("session_1", "user", "What is DBMS?")
        conversation_memory.save_message("session_1", "assistant", "A Database Management System.")

        history = conversation_memory.load_history("session_1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "What is DBMS?")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "A Database Management System.")
        self.assertIn("timestamp", history[0])

    def test_save_and_load_with_image_paths(self):
        conversation_memory.save_message(
            "session_img",
            "assistant",
            "Chart of yield",
            image_paths=["/path/to/img1.png", "/path/to/img1.png", "/path/to/img2.png"],
        )

        history = conversation_memory.load_history("session_img")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["image_paths"], ["/path/to/img1.png", "/path/to/img2.png"])

    def test_workflow_strips_recursive_history(self):
        recursive_workflow = {
            "intent": "factual",
            "history": [
                {"role": "user", "content": "old query"},
                {"role": "assistant", "content": "old answer", "workflow": {"intent": "factual"}},
            ],
            "timings": [{"stage": "search", "elapsed": 0.1}],
            "answer": "Test answer",
        }

        conversation_memory.save_message(
            "session_wf",
            "assistant",
            "Test answer",
            workflow=recursive_workflow,
        )

        history = conversation_memory.load_history("session_wf")
        self.assertEqual(len(history), 1)
        saved_wf = history[0]["workflow"]
        self.assertEqual(saved_wf["intent"], "factual")
        self.assertEqual(saved_wf["answer"], "Test answer")
        self.assertNotIn("history", saved_wf, "History must be stripped to prevent recursive explosion")

    def test_list_conversations_ordered_by_recency(self):
        conversation_memory.save_message("session_a", "user", "Hello A")
        conversation_memory.save_message("session_b", "user", "Hello B")
        # Update session_a to make it more recent
        conversation_memory.save_message("session_a", "assistant", "Reply A")

        sessions = conversation_memory.list_conversations()
        self.assertEqual(sessions[0], "session_a")
        self.assertIn("session_b", sessions)

    def test_clear_conversation(self):
        conversation_memory.save_message("session_del", "user", "To be deleted")
        self.assertEqual(len(conversation_memory.load_history("session_del")), 1)

        result = conversation_memory.clear_conversation("session_del")
        self.assertTrue(result)
        self.assertEqual(len(conversation_memory.load_history("session_del")), 0)

    def test_create_new_conversation(self):
        cid = conversation_memory.create_new_conversation()
        self.assertTrue(cid.startswith("conv_"))
        self.assertEqual(len(cid.split("_")), 3)

    def test_json_fallback_mode_when_patched_with_json_path(self):
        json_path = Path(self.temp_dir.name) / "conversations.json"
        with patch.object(conversation_memory, "MEMORY_FILE", json_path):
            conversation_memory.save_message("session_json", "user", "JSON test")
            conversation_memory.save_message("session_json", "assistant", "JSON reply")

            history = conversation_memory.load_history("session_json")
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["content"], "JSON test")

            sessions = conversation_memory.list_conversations()
            self.assertIn("session_json", sessions)

            cleared = conversation_memory.clear_conversation("session_json")
            self.assertTrue(cleared)
            self.assertEqual(len(conversation_memory.load_history("session_json")), 0)


if __name__ == "__main__":
    unittest.main()
