"""OpenRouter provider implementation.

OpenRouter is a unified API that provides access to 500+ language models from
multiple providers including OpenAI, Anthropic, Google, Meta, and others. It offers
intelligent routing, cost optimization, and provider fallbacks.

Environment variables:
  - OPENROUTER_API_KEY: OpenRouter API key (required)

Model naming follows the standard format, e.g.:
  - openai/gpt-5
  - anthropic/claude-sonnet-4.1
  - deepseek/deepseek-chat-v3.1

Website: https://openrouter.ai
All Models: https://openrouter.ai/models
Get your API Key here: https://openrouter.ai/settings/keys
"""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class OpenRouterAPI(OpenAICompatibleAPI):
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        # Remove provider prefix
        model_name_clean = model_name.replace("openrouter/", "", 1)

        base_url = base_url or self.DEFAULT_BASE_URL

        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Set the OPENROUTER_API_KEY environment variable. "
                "Get your API key at https://openrouter.ai/settings/keys"
            )

        # Add custom headers for OpenBench identification
        if "default_headers" not in model_args:
            model_args["default_headers"] = {}

        model_args["default_headers"].update(
            {
                "HTTP-Referer": "https://github.com/groq/openbench",
                "X-Title": "OpenBench",
            }
        )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="openrouter",
            service_base_url=self.DEFAULT_BASE_URL,
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
