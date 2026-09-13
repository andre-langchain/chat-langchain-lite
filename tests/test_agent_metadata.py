import os
import unittest
from unittest.mock import patch


os.environ.setdefault("LANGSMITH_API_KEY_GATEWAY", "test-key")

from agent.agent import _config
from utils.models import MODEL_CONFIG


class AgentMetadataTest(unittest.TestCase):
    def test_config_always_emits_trace_metadata(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHAT_LANGCHAIN_LITE_ENV", None)
            metadata = _config()["metadata"]

        self.assertTrue(metadata["thread_id"])
        self.assertEqual(metadata["user_id"], "anonymous")
        self.assertEqual(metadata["environment"], "development")
        self.assertEqual(metadata["model"], MODEL_CONFIG["model"])

    def test_config_preserves_supplied_metadata(self):
        metadata = _config("thread", "user", "staging")["metadata"]

        self.assertEqual(metadata["thread_id"], "thread")
        self.assertEqual(metadata["user_id"], "user")
        self.assertEqual(metadata["environment"], "staging")
        self.assertEqual(metadata["model"], MODEL_CONFIG["model"])


if __name__ == "__main__":
    unittest.main()
