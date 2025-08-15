"""Cohere provider implementation."""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class CohereAPI(OpenAICompatibleAPI):
    """Cohere provider - enterprise-ready language model infrastructure.

    Uses OpenAI-compatible API with Cohere-specific optimizations.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        # Extract model name without service prefix
        model_name_clean = model_name.replace("cohere/", "", 1)

        # Set defaults for Cohere
        base_url = base_url or os.environ.get(
            "COHERE_BASE_URL", "https://api.cohere.ai/compatibility/v1"
        )
        api_key = api_key or os.environ.get("COHERE_API_KEY")

        if not api_key:
            raise ValueError(
                "Cohere API key not found. Set COHERE_API_KEY environment variable."
            )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="cohere",
            service_base_url="https://api.cohere.ai/compatibility/v1",
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
