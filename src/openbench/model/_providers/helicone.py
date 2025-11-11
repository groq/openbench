"""Helicone AI Gateway provider implementation.

Helicone is an open-sourced AI gateway with observability included by default, providing unified access
to 100+ AI models from various providers including OpenAI, Anthropic, Google, and others.

Environment variables:
  - HELICONE_API_KEY: Helicone API key (required, unless passed directly via api_key parameter)

Model naming follows the model/provider format, e.g.:
  - helicone/gpt-4o (automatic provider routing)
  - helicone/claude-3-sonnet/anthropic (specific provider)
  - helicone/llama-3.1-70b/groq (specific provider)

Helicone-specific features can be configured via additional parameters:
  - cache_enabled: Enable/disable caching (boolean)
  - user_id: User identifier for request tracking
  - session_id: Session identifier for request grouping
  - custom_properties: Custom metadata for request tracking (dict)
  - rate_limit_policy: Rate limiting policy name
  - fallback_enabled: Enable/disable fallback providers (boolean)

Website: https://helicone.ai
Documentation: https://docs.helicone.ai
Models supported: https://helicone.ai/models
Get your API Key here: https://us.helicone.ai/settings/api-keys
"""

import os
from typing import Any, Dict

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class HeliconeAPI(OpenAICompatibleAPI):
    DEFAULT_BASE_URL = "https://ai-gateway.helicone.ai"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        cache_enabled: bool | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        custom_properties: Dict[str, Any] | None = None,
        rate_limit_policy: str | None = None,
        fallback_enabled: bool | None = None,
        **model_args: Any,
    ) -> None:
        # Remove helicone prefix and extract model/provider
        model_name_clean = model_name.replace("helicone/", "", 1)

        # Helicone handles provider routing through model/provider format (e.g. "gpt-4o" or "gpt-4o/azure")
        helicone_model_name = model_name_clean

        base_url = base_url or self.DEFAULT_BASE_URL

        # Get Helicone API key
        helicone_api_key = api_key or os.environ.get("HELICONE_API_KEY")
        if not helicone_api_key:
            raise ValueError(
                "Helicone API key not found. Set the HELICONE_API_KEY environment variable. "
                "Get your API key at https://us.helicone.ai/settings/api-keys"
            )

        # Set up custom headers for Helicone
        if "default_headers" not in model_args:
            model_args["default_headers"] = {}

        # Add Helicone authentication header
        model_args["default_headers"]["Helicone-Auth"] = f"Bearer {helicone_api_key}"

        # Add custom identification headers
        model_args["default_headers"].update(
            {
                "Helicone-Property-Source": "openbench",
                "Helicone-Property-Repository": "https://github.com/groq/openbench",
            }
        )

        # Build Helicone-specific properties
        helicone_properties = {}

        if user_id is not None:
            helicone_properties["user-id"] = user_id
        if session_id is not None:
            helicone_properties["session-id"] = session_id
        if custom_properties:
            helicone_properties.update(custom_properties)

        # Add properties as headers
        for key, value in helicone_properties.items():
            model_args["default_headers"][f"Helicone-Property-{key}"] = str(value)

        # Add cache control header if specified
        if cache_enabled is not None:
            model_args["default_headers"]["Helicone-Cache-Enabled"] = str(
                cache_enabled
            ).lower()

        # Add rate limit policy if specified
        if rate_limit_policy is not None:
            model_args["default_headers"]["Helicone-RateLimit-Policy"] = (
                rate_limit_policy
            )

        # Store extra body parameters for Helicone features
        self._extra_body = {}
        if fallback_enabled is not None:
            self._extra_body["helicone_fallback_enabled"] = fallback_enabled

        super().__init__(
            model_name=helicone_model_name,
            base_url=base_url,
            api_key=helicone_api_key,  # Use Helicone key as the main API key
            config=config,
            service="helicone",
            service_base_url=self.DEFAULT_BASE_URL,
            **model_args,
        )

        # Inject Helicone-specific parameters into chat completion requests
        if self._extra_body:
            original_create = self.client.chat.completions.create

            def create_with_helicone_features(**kwargs):
                # Merge Helicone parameters with any existing extra_body
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {}
                if kwargs["extra_body"] is None:
                    kwargs["extra_body"] = {}
                kwargs["extra_body"].update(self._extra_body)
                return original_create(**kwargs)

            # Replace the create method
            setattr(
                self.client.chat.completions, "create", create_with_helicone_features
            )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
