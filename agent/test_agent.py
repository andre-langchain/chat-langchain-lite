import importlib
import os
import unittest
from unittest.mock import Mock, patch

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class InvokeAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"LANGSMITH_API_KEY_GATEWAY": "test-key"}), patch(
            "dotenv.load_dotenv"
        ), patch("context.get_prompt", return_value="Test prompt"):
            cls.agent = importlib.import_module("agent.agent")

    def invoke_with_messages(self, messages):
        with patch.object(self.agent, "build_agent", return_value=Mock()) as build_agent:
            build_agent.return_value.invoke.return_value = {"messages": messages}
            return self.agent.invoke_agent("Explain LangGraph")

    def test_truncated_response_is_marked(self):
        messages = [
            HumanMessage(content="Explain LangGraph"),
            ToolMessage(content="Guide", name="get_setup_guide", tool_call_id="setup"),
            AIMessage(content="An unfinished answer", response_metadata={"stop_reason": "max_tokens"}),
        ]

        result = self.invoke_with_messages(messages)

        self.assertEqual(
            result["output"],
            "An unfinished answer\n\n_(Response truncated at the output limit.)_",
        )
        self.assertEqual(result["tools_called"], ["get_setup_guide"])
        self.assertEqual(result["messages"], messages)
        self.assertEqual(messages[-1].content, "An unfinished answer")

    def test_completed_response_is_unchanged(self):
        answer = "Example:\n```python\nprint('complete')\n```"
        for metadata in ({"stop_reason": "end_turn"}, {}, {"stop_reason": "stop_sequence"}):
            with self.subTest(metadata=metadata):
                result = self.invoke_with_messages([
                    AIMessage(content="Earlier cutoff", response_metadata={"stop_reason": "max_tokens"}),
                    AIMessage(content=answer, response_metadata=metadata),
                ])
                self.assertEqual(result["output"], answer)

    def test_truncation_notice_is_outside_code_fences(self):
        cases = [
            ("```python\nprint(", "\n```"),
            ("~~~python\nprint(", "\n~~~"),
            ("````python\n```\nprint(", "\n````"),
            ("```python\nprint('done')\n```", ""),
            ("  ```python\nprint(", "\n```"),
            ("```python\nprint('done')\n```\n~~~python\nprint(", "\n~~~"),
        ]
        for answer, closing_fence in cases:
            with self.subTest(answer=answer):
                result = self.invoke_with_messages([
                    AIMessage(content=answer, response_metadata={"stop_reason": "max_tokens"}),
                ])
                self.assertEqual(
                    result["output"],
                    answer + closing_fence + "\n\n_(Response truncated at the output limit.)_",
                )

    def test_long_response_passes_through_agent_graph(self):
        answer = "\n".join(
            f"Step {step}: configure and invoke the LangGraph workflow." for step in range(100)
        )
        self.assertGreater(len(answer.split()), 300)
        for stop_reason in ("end_turn", "max_tokens"):
            with self.subTest(stop_reason=stop_reason):
                graph = create_agent(model=FakeMessagesListChatModel(responses=[
                    AIMessage(content=answer, response_metadata={"stop_reason": stop_reason}),
                ]))

                with patch.object(self.agent, "build_agent", return_value=graph), patch.dict(
                    os.environ, {"LANGSMITH_TRACING": "false"}
                ):
                    result = self.agent.invoke_agent("Give a detailed technical walkthrough")

                expected = answer
                if stop_reason == "max_tokens":
                    expected += "\n\n_(Response truncated at the output limit.)_"
                self.assertEqual(result["output"], expected)
