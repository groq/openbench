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

Provider routing parameters can be specified to control which providers are used:
  - only: Restrict to specific providers (e.g., only=groq or only=cerebras,openai)
  - order: Provider priority order (e.g., order=openai,anthropic)
  - allow_fallbacks: Enable/disable fallback providers (boolean)
  - ignore: Providers to skip (e.g., ignore=cerebras,fireworks)
  - sort: Sort providers by "price" or "throughput"
  - max_price: Maximum price limits (e.g., max_price={"completion": 0.01})
  - quantizations: Filter by quantization levels (e.g., quantizations=int4,int8)
  - require_parameters: Require parameter support (boolean)
  - data_collection: Data collection setting ("allow" or "deny")

Website: https://openrouter.ai
All Models: https://openrouter.ai/models
Get your API Key here: https://openrouter.ai/settings/keys
Provider Routing Docs: https://openrouter.ai/docs/features/provider-routing
"""

from typing import Any, List, Dict

from inspect_ai.model._providers.openrouter import (
    OpenRouterAPI as InspectAIOpenRouterAPI,
)
from inspect_ai.model import GenerateConfig


class OpenRouterAPI(InspectAIOpenRouterAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        only: List[str] | str | None = None,
        order: List[str] | str | None = None,
        allow_fallbacks: bool | None = None,
        ignore: List[str] | str | None = None,
        sort: str | None = None,
        max_price: Dict[str, float] | None = None,
        quantizations: List[str] | str | None = None,
        require_parameters: bool | None = None,
        data_collection: str | None = None,
        **model_args: Any,
    ) -> None:
        # Remove provider prefix
        model_name_clean = model_name.replace("openrouter/", "", 1)

        # Build provider routing object from parameters
        provider_params: Dict[str, Any] = {}

        # Handle list/string parameters that can be passed as comma-separated strings
        def _parse_list_param(param: List[str] | str | None) -> List[str] | None:
            if param is None:
                return None
            if isinstance(param, str):
                return [p.strip() for p in param.split(",")]
            return param

        if only is not None:
            provider_params["only"] = _parse_list_param(only)
        if order is not None:
            provider_params["order"] = _parse_list_param(order)
        if ignore is not None:
            provider_params["ignore"] = _parse_list_param(ignore)
        if quantizations is not None:
            provider_params["quantizations"] = _parse_list_param(quantizations)
        if allow_fallbacks is not None:
            provider_params["allow_fallbacks"] = allow_fallbacks
        if sort is not None:
            provider_params["sort"] = sort
        if max_price is not None:
            provider_params["max_price"] = max_price
        if require_parameters is not None:
            provider_params["require_parameters"] = require_parameters
        if data_collection is not None:
            provider_params["data_collection"] = data_collection

        # Pass provider routing to upstream via model_args (upstream expects "provider" key)
        if provider_params:
            model_args["provider"] = provider_params

        # Add custom headers for openbench identification
        if "default_headers" not in model_args:
            model_args["default_headers"] = {}

        model_args["default_headers"].update(
            {
                "HTTP-Referer": "https://github.com/groq/openbench",
                "X-Title": "openbench",
            }
        )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
