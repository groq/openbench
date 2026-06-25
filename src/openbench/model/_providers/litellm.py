"""LiteLLM AI gateway provider implementation."""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class LiteLLMAPI(OpenAICompatibleAPI):
    """LiteLLM AI gateway provider - access 100+ LLM providers through one interface.

    Uses OpenAI-compatible API pointed at a LiteLLM proxy. The model name
    is passed through as-is, so use whatever model identifiers your LiteLLM
    config defines (e.g. ``litellm/anthropic/claude-sonnet-4-6``).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        model_name_clean = model_name.replace("litellm/", "", 1)

        base_url = base_url or os.environ.get(
            "LITELLM_BASE_URL", "http://localhost:4000/v1"
        )
        api_key = api_key or os.environ.get("LITELLM_API_KEY")

        if not api_key:
            raise ValueError(
                "LiteLLM API key not found. Set LITELLM_API_KEY environment variable."
            )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="litellm",
            service_base_url="http://localhost:4000/v1",
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
