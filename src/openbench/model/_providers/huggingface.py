"""Hugging Face Inference Providers (OpenAI-compatible) provider.

Uses the Hugging Face Inference Providers router over the OpenAI-compatible
API, as documented in the GPT OSS guide:

Reference: https://huggingface.co/docs/inference-providers/en/guides/gpt-oss

Environment variables:
  - HF_TOKEN: Hugging Face access token used as Bearer token
  - HF_ROUTER_BASE_URL: Optional override for base URL (defaults to
    https://router.huggingface.co/v1)

Model naming follows the HF router format, e.g.:
  - openai/gpt-oss-120b:cerebras
  - openai/gpt-oss-120b:fireworks-ai
"""

from __future__ import annotations

import os
from typing import Any

from inspect_ai.model import (  # type: ignore[import-not-found]
    GenerateConfig,
)
from inspect_ai.model._providers.openai_compatible import (  # type: ignore[import-not-found]
    OpenAICompatibleAPI,
)


class HFInferenceProvidersAPI(OpenAICompatibleAPI):
    """Hugging Face Inference Providers API."""

    DEFAULT_BASE_URL = "https://router.huggingface.co/v1"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        # Remove provider prefix
        model_name_clean = model_name.replace("huggingface/", "", 1)

        base_url = base_url or os.environ.get("HF_BASE_URL") or self.DEFAULT_BASE_URL
        api_key = api_key or os.environ.get("HF_TOKEN")

        if not api_key:
            raise ValueError(
                "HF_TOKEN not set. Get a token from your HF settings page."
            )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="huggingface",
            service_base_url=self.DEFAULT_BASE_URL,
            **model_args,
        )

    def service_model_name(self) -> str:
        return self.model_name
