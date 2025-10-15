"""Vercel AI Gateway provider implementation.

The AI Gateway provides OpenAI-compatible API endpoints, letting you use multiple
AI providers through a familiar interface. The AI Gateway can route requests across
multiple AI providers for better reliability and performance.

Environment variables:
  - AI_GATEWAY_API_KEY: AI Gateway API key (required)
  - AI_GATEWAY_BASE_URL: Override the default base URL (defaults to
    https://ai-gateway.vercel.sh/v1)

Model naming follows the creator/model format, e.g.:
  - anthropic/claude-sonnet-4
  - openai/gpt-4.1-mini
  - meta/llama-3.3-70b-instruct

Website: https://vercel.com/ai-gateway
Reference: https://vercel.com/docs/ai-gateway/openai-compatible-api
"""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class VercelAPI(OpenAICompatibleAPI):
    """Vercel AI Gateway provider - OpenAI-compatible API with multi-provider routing."""

    DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v1"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        # Remove provider prefix if present
        # Result is in creator/model format
        model_name_clean = model_name.replace("vercel/", "", 1)

        base_url = (
            base_url or os.environ.get("AI_GATEWAY_BASE_URL") or self.DEFAULT_BASE_URL
        )
        api_key = (
            api_key
            or os.environ.get("AI_GATEWAY_API_KEY")
            or os.environ.get("VERCEL_OIDC_TOKEN")
        )

        if not api_key:
            raise ValueError(
                "Vercel AI Gateway API key not found. Set the AI_GATEWAY_API_KEY environment variable."
            )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="vercel",
            service_base_url=self.DEFAULT_BASE_URL,
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
