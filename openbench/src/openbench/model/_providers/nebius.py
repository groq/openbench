"""Nebius AI provider implementation."""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class NebiusAPI(OpenAICompatibleAPI):
    """Nebius AI provider - OpenAI-compatible inference.

    Uses OpenAI-compatible API with Nebius Studio endpoints.
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
        model_name_clean = model_name.replace("nebius/", "", 1)

        # Set defaults for Nebius
        base_url = base_url or os.environ.get(
            "NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1"
        )
        api_key = api_key or os.environ.get("NEBIUS_API_KEY")

        if not api_key:
            raise ValueError(
                "Nebius API key not found. Set NEBIUS_API_KEY environment variable."
            )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="nebius",
            service_base_url="https://api.studio.nebius.com/v1",
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
