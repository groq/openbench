"""W&B Inference provider implementation.

Access open-source foundation models through W&B Inference and an OpenAI-compatible API.
"""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class WandBInferenceAPI(OpenAICompatibleAPI):
    """W&B Inference provider - OpenAI-compatible inference.

    Uses OpenAI-compatible API with W&B Inference-specific configuration.
    Reference: https://docs.wandb.ai/guides/weave/reference/wandb_weave_inference
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
        model_name_clean = model_name.replace("wandb/", "", 1)

        # Set defaults for W&B Inference
        base_url = base_url or os.environ.get(
            "WANDB_INFERENCE_BASE_URL", "https://api.inference.wandb.ai/v1"
        )
        api_key = api_key or os.environ.get("WANDB_API_KEY")

        if not api_key:
            raise ValueError(
                "W&B API key not found. Set WANDB_API_KEY environment variable."
            )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="wandb",
            service_base_url="https://api.inference.wandb.ai/v1",
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
