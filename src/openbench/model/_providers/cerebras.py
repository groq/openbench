"""Cerebras AI provider implementation."""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class CerebrasAPI(OpenAICompatibleAPI):
    """Cerebras AI provider - high-performance inference.

    Uses OpenAI-compatible API with Cerebras-specific optimizations.
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
        model_name_clean = model_name.replace("cerebras/", "", 1)

        # Set defaults for Cerebras
        base_url = base_url or os.environ.get(
            "CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"
        )
        api_key = api_key or os.environ.get("CEREBRAS_API_KEY")

        if not api_key:
            raise ValueError(
                "Cerebras API key not found. Set CEREBRAS_API_KEY environment variable."
            )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="cerebras",
            service_base_url="https://api.cerebras.ai/v1",
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
