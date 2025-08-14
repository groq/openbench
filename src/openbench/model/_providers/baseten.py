"""Baseten AI provider implementation."""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class BasetenAPI(OpenAICompatibleAPI):
    """Baseten AI provider - OpenAI-compatible inference.

    Uses OpenAI-compatible API with Baseten-specific configuration.
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
        model_name_clean = model_name.replace("baseten/", "", 1)

        # Set defaults for Baseten
        base_url = base_url or os.environ.get(
            "BASETEN_BASE_URL", "https://inference.baseten.co/v1"
        )
        api_key = api_key or os.environ.get("BASETEN_API_KEY")

        if not api_key:
            raise ValueError(
                "Baseten API key not found. Set BASETEN_API_KEY environment variable."
            )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="baseten",
            service_base_url="https://inference.baseten.co/v1",
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
