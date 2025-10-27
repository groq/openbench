"""Unit tests for vLLM provider timeout configuration."""

from unittest.mock import patch

from inspect_ai.model import GenerateConfig

from openbench.model._providers.vllm import VLLMAPI


class TestVLLMProviderTimeout:
    """Test vLLM provider timeout configuration."""

    def test_timeout_from_config(self):
        """Client is created with timeout from GenerateConfig."""
        with patch(
            "openbench.model._providers.vllm.OpenAIAsyncHttpxClient"
        ) as mock_client:
            # Avoid creating a real AsyncOpenAI client
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                config = GenerateConfig(timeout=300)

                VLLMAPI(
                    model_name="vllm/test-model",
                    base_url="http://localhost:8000/v1",
                    api_key="test-key",
                    config=config,
                )

                mock_client.assert_called_once()
                kwargs = mock_client.call_args.kwargs
                assert "timeout" in kwargs
                # httpx.Timeout is constructed below; we just ensure we passed one through
                timeout_obj = kwargs["timeout"]
                # httpx.Timeout exposes read/connect/write attributes
                assert getattr(timeout_obj, "read", None) == 300

    def test_no_timeout_when_not_in_config(self):
        """No client override is constructed when timeout unset."""
        with patch(
            "openbench.model._providers.vllm.OpenAIAsyncHttpxClient"
        ) as mock_client:
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                config = GenerateConfig()  # No timeout set

                VLLMAPI(
                    model_name="vllm/test-model",
                    base_url="http://localhost:8000/v1",
                    api_key="test-key",
                    config=config,
                )

                # We don't build a custom client when timeout is not provided
                mock_client.assert_not_called()

    def test_no_timeout_when_none_in_config(self):
        """No client override is constructed when timeout is None."""
        with patch(
            "openbench.model._providers.vllm.OpenAIAsyncHttpxClient"
        ) as mock_client:
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                config = GenerateConfig(timeout=None)

                VLLMAPI(
                    model_name="vllm/test-model",
                    base_url="http://localhost:8000/v1",
                    api_key="test-key",
                    config=config,
                )

                mock_client.assert_not_called()

    def test_httpx_timeout_object_creation(self):
        """httpx.Timeout is used to construct the client timeout."""
        with (
            patch(
                "openbench.model._providers.vllm.OpenAIAsyncHttpxClient"
            ) as mock_client,
            patch("httpx.Timeout") as mock_timeout,
            patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"),
        ):
            config = GenerateConfig(timeout=180)

            VLLMAPI(
                model_name="vllm/test-model",
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                config=config,
            )

            mock_timeout.assert_called_once_with(timeout=180)
            mock_client.assert_called_once()
            assert (
                mock_client.call_args.kwargs.get("timeout") == mock_timeout.return_value
            )

    def test_timeout_zero_value(self):
        """timeout=0 is forwarded correctly to the client builder."""
        with patch(
            "openbench.model._providers.vllm.OpenAIAsyncHttpxClient"
        ) as mock_client:
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                config = GenerateConfig(timeout=0)

                VLLMAPI(
                    model_name="vllm/test-model",
                    base_url="http://localhost:8000/v1",
                    api_key="test-key",
                    config=config,
                )

                mock_client.assert_called_once()
                timeout_obj = mock_client.call_args.kwargs.get("timeout")
                assert getattr(timeout_obj, "read", None) == 0

    def test_multiple_client_instances(self):
        """Different instances use their own configured timeouts."""
        with patch(
            "openbench.model._providers.vllm.OpenAIAsyncHttpxClient"
        ) as mock_client:
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                VLLMAPI(
                    model_name="vllm/test-model-1",
                    base_url="http://localhost:8000/v1",
                    api_key="test-key",
                    config=GenerateConfig(timeout=100),
                )

                VLLMAPI(
                    model_name="vllm/test-model-2",
                    base_url="http://localhost:8000/v1",
                    api_key="test-key",
                    config=GenerateConfig(timeout=200),
                )

                assert mock_client.call_count == 2
                call1, call2 = mock_client.call_args_list
                assert getattr(call1.kwargs.get("timeout"), "read", None) == 100
                assert getattr(call2.kwargs.get("timeout"), "read", None) == 200
