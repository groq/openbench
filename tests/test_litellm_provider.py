"""Unit tests for LiteLLM AI gateway provider."""

import os
from unittest.mock import patch

import pytest

from openbench.model._providers.litellm import LiteLLMAPI


class TestLiteLLMProvider:
    """Test LiteLLM provider initialization and configuration."""

    def test_raises_without_api_key(self):
        """Provider raises ValueError when no API key is available."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="LITELLM_API_KEY"):
                with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                    LiteLLMAPI(model_name="litellm/gpt-4o")

    def test_strips_service_prefix_from_model_name(self):
        """The litellm/ prefix is stripped before passing to the base class."""
        with patch.dict(os.environ, {"LITELLM_API_KEY": "sk-test"}):
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                provider = LiteLLMAPI(
                    model_name="litellm/anthropic/claude-sonnet-4-6",
                    api_key="sk-test",
                    base_url="http://localhost:4000/v1",
                )
                assert provider.model_name == "anthropic/claude-sonnet-4-6"

    def test_uses_env_api_key(self):
        """Provider reads LITELLM_API_KEY from environment."""
        with patch.dict(os.environ, {"LITELLM_API_KEY": "sk-from-env"}):
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                provider = LiteLLMAPI(
                    model_name="litellm/gpt-4o",
                    base_url="http://localhost:4000/v1",
                )
                assert provider.api_key == "sk-from-env"

    def test_explicit_api_key_overrides_env(self):
        """Explicit api_key parameter takes precedence over env var."""
        with patch.dict(os.environ, {"LITELLM_API_KEY": "sk-from-env"}):
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                provider = LiteLLMAPI(
                    model_name="litellm/gpt-4o",
                    api_key="sk-explicit",
                    base_url="http://localhost:4000/v1",
                )
                assert provider.api_key == "sk-explicit"

    def test_default_base_url(self):
        """Provider defaults to localhost:4000 when no base URL is set."""
        with patch.dict(os.environ, {"LITELLM_API_KEY": "sk-test"}, clear=True):
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                provider = LiteLLMAPI(model_name="litellm/gpt-4o")
                assert provider.base_url == "http://localhost:4000/v1"

    def test_custom_base_url_from_env(self):
        """Provider reads LITELLM_BASE_URL from environment."""
        with patch.dict(
            os.environ,
            {
                "LITELLM_API_KEY": "sk-test",
                "LITELLM_BASE_URL": "https://proxy.example.com/v1",
            },
        ):
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                provider = LiteLLMAPI(model_name="litellm/gpt-4o")
                assert provider.base_url == "https://proxy.example.com/v1"

    def test_service_model_name_returns_clean_name(self):
        """service_model_name() returns the model name without prefix."""
        with patch.dict(os.environ, {"LITELLM_API_KEY": "sk-test"}):
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                provider = LiteLLMAPI(
                    model_name="litellm/openai/gpt-4o",
                    base_url="http://localhost:4000/v1",
                )
                assert provider.service_model_name() == "openai/gpt-4o"
