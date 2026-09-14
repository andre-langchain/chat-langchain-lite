"""Centralized model initialization"""

import os
from dotenv import load_dotenv
# Search upward from the cwd rather than hard-coding "../.env", which resolved
# outside the repo and silently loaded nothing. Matches every other module.
load_dotenv(override=True)

from langchain.chat_models import init_chat_model

# --- Default: OpenAI, direct ---
# model = init_chat_model("openai:gpt-4.1-mini")

# --- OpenAI via the LangSmith LLM Gateway ---
# Routes every model call through the LangSmith Gateway so that workspace
# policies (PII / secrets / allow-lists / cost caps) are enforced.
# MODEL_CONFIG is the single source the frontend's Gateway pane reads.
MODEL_CONFIG = {
    "model": "claude-sonnet-4-6",
    "provider": "anthropic",
    "base_url": "https://gateway.smith.langchain.com/anthropic",
}
_gateway_api_key = os.environ.get("LANGSMITH_API_KEY_GATEWAY")
if not _gateway_api_key:
    raise RuntimeError(
        "LANGSMITH_API_KEY_GATEWAY is not set. This is the key the LangSmith "
        "LLM Gateway authenticates model calls with, and it is separate from "
        "LANGSMITH_API_KEY (tracing). Locally: add it to .env (see "
        ".env.example). In GitHub Actions: add it as a repository secret and "
        "map it into the job's env: block in .github/workflows/evals.yml."
    )

model = init_chat_model(
    model=MODEL_CONFIG["model"],
    model_provider=MODEL_CONFIG["provider"],
    base_url=MODEL_CONFIG["base_url"],
    api_key=_gateway_api_key,
    # 300 tokens truncated mid-sentence and mid-code-block; let the system prompt steer brevity.
    max_tokens=4096,
    temperature=0,
)

# --- Anthropic ---
# model = init_chat_model("anthropic:claude-sonnet-4-5")

# --- Azure OpenAI ---
# from langchain_openai import AzureChatOpenAI
# model = AzureChatOpenAI(azure_deployment="gpt-4.1-mini", streaming=True)

# --- AWS Bedrock ---
# from langchain_aws import ChatBedrockConverse
# model = ChatBedrockConverse(
#     provider="anthropic",
#     model_id="anthropic.claude-sonnet-4-20250514-v1:0",
# )
