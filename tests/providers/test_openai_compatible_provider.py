"""Test the OpenAI-compatible provider implementation."""

import os
import pytest
from unittest.mock import patch

from openbench.model._providers.openai_compatible import OpenAICompatibleProviderAPI
from inspect_ai.model import GenerateConfig


class TestOpenAICompatibleProviderAPI:
    """Test suite for OpenAICompatibleProviderAPI provider."""

    def test_provider_initialization_with_defaults(self):
        """Test provider initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            # Should use default base_url and dummy API key
            provider = OpenAICompatibleProviderAPI(
                model_name="openai_compatible/test-model", config=GenerateConfig()
            )

            assert provider.model_name == "test-model"
            assert provider.service_model_name() == "test-model"

    def test_provider_initialization_with_env_vars(self):
        """Test provider initialization with environment variables."""
        test_env = {
            "OPENAI_COMPATIBLE_BASE_URL": "http://localhost:8080/v1",
            "OPENAI_COMPATIBLE_API_KEY": "test-key-123",
        }

        with patch.dict(os.environ, test_env, clear=True):
            provider = OpenAICompatibleProviderAPI(
                model_name="openai_compatible/custom-model", config=GenerateConfig()
            )

            assert provider.model_name == "custom-model"
            assert provider.service_model_name() == "custom-model"

    def test_provider_initialization_with_explicit_params(self):
        """Test provider initialization with explicit parameters."""
        provider = OpenAICompatibleProviderAPI(
            model_name="openai_compatible/explicit-model",
            base_url="http://custom.endpoint.com/v1",
            api_key="explicit-key",
            config=GenerateConfig(),
        )

        assert provider.model_name == "explicit-model"
        assert provider.service_model_name() == "explicit-model"

    def test_model_name_prefix_removal(self):
        """Test that provider prefix is correctly removed from model name."""
        test_cases = [
            ("openai_compatible/llama-2-7b", "llama-2-7b"),
            ("openai_compatible/gpt-3.5-turbo", "gpt-3.5-turbo"),
            ("custom-model-without-prefix", "custom-model-without-prefix"),
            ("openai_compatible/", ""),  # Edge case
        ]

        for input_name, expected_output in test_cases:
            provider = OpenAICompatibleProviderAPI(
                model_name=input_name,
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                config=GenerateConfig(),
            )

            assert provider.model_name == expected_output
            assert provider.service_model_name() == expected_output

    def test_environment_variable_defaults(self):
        """Test default environment variable values."""
        with patch.dict(os.environ, {}, clear=True):
            # Test default base_url when no env var is set
            provider = OpenAICompatibleProviderAPI(
                model_name="test-model", config=GenerateConfig()
            )

            # Can't directly access base_url from the provider, but we can verify
            # that it was initialized without errors with the default values
            assert provider.model_name == "test-model"

    def test_api_key_fallback(self):
        """Test API key fallback to dummy key when not provided."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise an error even without API key
            provider = OpenAICompatibleProviderAPI(
                model_name="test-model", config=GenerateConfig()
            )

            assert provider.model_name == "test-model"

    def test_custom_config_parameter(self):
        """Test that custom GenerateConfig is accepted."""
        custom_config = GenerateConfig(max_tokens=1000, temperature=0.7, top_p=0.9)

        provider = OpenAICompatibleProviderAPI(
            model_name="test-model", config=custom_config
        )

        assert provider.model_name == "test-model"

    def test_additional_model_args(self):
        """Test that additional model arguments are accepted."""
        provider = OpenAICompatibleProviderAPI(
            model_name="test-model",
            config=GenerateConfig(),
            timeout=60.0,
            max_retries=3,
        )

        assert provider.model_name == "test-model"

    @pytest.fixture(autouse=True)
    def clean_environment(self):
        """Clean environment variables before each test."""
        # Store original environment
        original_env = os.environ.copy()

        # Clean specific env vars
        env_vars_to_clean = ["OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_API_KEY"]

        for var in env_vars_to_clean:
            os.environ.pop(var, None)

        yield

        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)
