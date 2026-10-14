import importlib
import os
import unittest
from unittest.mock import Mock, patch

from anthropic.types import Message, TextBlock, Usage
from langchain.agents.middleware import AgentMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler


class ModelMetadataCapture(BaseCallbackHandler):
    def __init__(self):
        self.root_metadata = []
        self.llm_metadata = []

    def on_chain_start(self, serialized, inputs, *, parent_run_id=None, **kwargs):
        if parent_run_id is None:
            self.root_metadata.append(kwargs.get("metadata", {}))

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.llm_metadata.append(kwargs.get("metadata", {}))


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "LANGSMITH_API_KEY_GATEWAY": "test-gateway-key",
            "LANGSMITH_TRACING": "false",
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        dotenv = patch("dotenv.load_dotenv")
        dotenv.start()
        self.addCleanup(dotenv.stop)
        from utils import models
        self.models = importlib.reload(models)

    def _load_agent(self):
        with patch("context.get_prompt", return_value="Test system prompt"):
            from agent import agent
            return importlib.reload(agent)

    def test_default_model_and_ui_configuration_select_haiku(self):
        model = self.models.get_model()
        self.assertIsInstance(model, ChatAnthropic)
        self.assertEqual(model.model, "claude-haiku-4-5-20251001")
        self.assertEqual(self.models.MODEL_CONFIG["model"], model.model)
        self.assertEqual(self.models.MODEL_CONFIG["provider"], "anthropic")
        self.assertEqual(model.anthropic_api_url, self.models.MODEL_CONFIG["base_url"])
        self.assertEqual(model.anthropic_api_key.get_secret_value(), "test-gateway-key")
        self.assertEqual(model.max_tokens, 300)
        self.assertEqual(model.temperature, 0)

    def test_environment_overrides_default_and_ui_configuration(self):
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = "claude-sonnet-4-6"
        importlib.reload(self.models)
        self.assertEqual(self.models.get_model().model, "claude-sonnet-4-6")
        self.assertEqual(self.models.MODEL_CONFIG["model"], "claude-sonnet-4-6")

    def test_explicit_model_overrides_environment(self):
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = "claude-sonnet-4-6"
        model = self.models.get_model("claude-haiku-4-5-20251001")
        self.assertEqual(model.model, "claude-haiku-4-5-20251001")

    def test_empty_environment_uses_default(self):
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = ""
        importlib.reload(self.models)
        self.assertEqual(self.models.get_model().model, "claude-haiku-4-5-20251001")
        self.assertEqual(self.models.MODEL_CONFIG["model"], "claude-haiku-4-5-20251001")

    def test_module_level_model_remains_available(self):
        self.assertIsInstance(self.models.model, ChatAnthropic)
        self.assertEqual(self.models.model.model, self.models.MODEL_CONFIG["model"])
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = "claude-sonnet-4-6"
        importlib.reload(self.models)
        self.assertEqual(self.models.model.model, "claude-sonnet-4-6")

    def test_factory_reads_environment_after_import(self):
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = "claude-sonnet-4-6"
        self.assertEqual(self.models.get_model().model, "claude-sonnet-4-6")
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = "claude-haiku-4-5-20251001"
        self.assertEqual(self.models.get_model().model, "claude-haiku-4-5-20251001")

    def test_build_agent_passes_selected_model_without_changing_prompt_or_tools(self):
        agent = self._load_agent()
        with (
            patch.object(agent, "_model_id", return_value="claude-sonnet-4-6"),
            patch.object(agent, "get_model", wraps=self.models.get_model) as factory,
            patch.object(agent, "create_agent") as create_agent,
            patch.object(agent, "_readonly_context_hub_fs", return_value="middleware"),
        ):
            self.assertIs(agent.build_agent(), create_agent.return_value)
        factory.assert_called_once_with("claude-sonnet-4-6")
        arguments = create_agent.call_args.kwargs
        self.assertEqual(arguments["model"].model, "claude-sonnet-4-6")
        self.assertIs(arguments["tools"], agent.TOOLS)
        self.assertEqual(arguments["system_prompt"], agent.SYSTEM_PROMPT)
        self.assertEqual(arguments["middleware"], ["middleware"])

    def test_baseline_models_match_root_and_chat_anthropic_metadata(self):
        agent = self._load_agent()
        from scripts.run_evals import run_agent_on_example
        from scripts.setup import _BASELINE_MODELS

        for model_id, label in _BASELINE_MODELS:
            with self.subTest(model=model_id, baseline=label):
                os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = model_id
                capture = ModelMetadataCapture()
                config = agent._config()
                config["callbacks"] = [capture]
                response = Message(
                    id="msg_test",
                    type="message",
                    role="assistant",
                    model=model_id,
                    content=[TextBlock(type="text", text="Test response")],
                    stop_reason="end_turn",
                    stop_sequence=None,
                    usage=Usage(input_tokens=10, output_tokens=5),
                )
                raw_response = Mock(wraps=response, parse=Mock(return_value=response), headers={})
                with (
                    patch("anthropic.resources.messages.Messages.create", return_value=raw_response) as request,
                    patch.object(agent, "_readonly_context_hub_fs", return_value=AgentMiddleware()),
                    patch.object(agent, "_config", return_value=config),
                ):
                    result = run_agent_on_example({"question": "What is LangChain?"})
                self.assertEqual(result["output"], "Test response")
                request.assert_called_once()
                self.assertEqual(request.call_args.kwargs["model"], model_id)
                self.assertEqual(len(capture.root_metadata), 1)
                self.assertEqual(capture.root_metadata[0]["model"], model_id)
                self.assertEqual(len(capture.llm_metadata), 1)
                self.assertEqual(capture.llm_metadata[0]["ls_model_name"], model_id)
                self.assertEqual(capture.llm_metadata[0]["model"], model_id)


if __name__ == "__main__":
    unittest.main()
