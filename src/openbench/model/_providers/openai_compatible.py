"""OpenAI-compatible provider implementation."""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class OpenAICompatibleProviderAPI(OpenAICompatibleAPI):
    """OpenAI-compatible provider for custom endpoints.

    Supports any OpenAI-compatible API endpoint, including:
    - llama.cpp server
    - tabbyAPI
    - Any other OpenAI-compatible endpoint


    Uses OpenAI-compatible API with configurable base URL and optional API key.
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
        model_name_clean = model_name.replace("openai_compatible/", "", 1)

        # Set defaults for OpenAI-compatible endpoint
        base_url = base_url or os.environ.get(
            "OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1"
        )

        # API key is optional for local setups (like vLLM)
        # Prioritize OPENAI_COMPATIBLE_API_KEY, only fallback to OPENAI_API_KEY if not set
        api_key = api_key or os.environ.get(
            "OPENAI_COMPATIBLE_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy-key")
        )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="openai_compatible",
            service_base_url="http://localhost:8000/v1",
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
