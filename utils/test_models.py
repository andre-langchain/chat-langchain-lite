import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


class ModelConfigurationTests(unittest.TestCase):
    def test_output_limit_and_temperature(self):
        with patch.dict(os.environ, {"LANGSMITH_API_KEY_GATEWAY": "test-key"}), patch(
            "dotenv.load_dotenv"
        ):
            model = runpy.run_path(str(Path(__file__).with_name("models.py")))["model"]

        self.assertEqual(model.max_tokens, 4096)
        self.assertEqual(model.temperature, 0)
