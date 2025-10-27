"""Unit tests for Groq provider timeout configuration."""

from unittest.mock import patch
from inspect_ai.model import GenerateConfig

from openbench.model._providers.groq import GroqAPI


class TestGroqProviderTimeout:
    """Test Groq provider timeout configuration."""

    def test_timeout_from_config(self):
        """Test that timeout is properly set from GenerateConfig."""
        # Mock the httpx.AsyncClient to avoid actual HTTP calls
        with patch("httpx.AsyncClient") as mock_client:
            with patch("openbench.model._providers.groq.AsyncGroq"):
                # Create a config with a specific timeout
                config = GenerateConfig(timeout=300)  # 5 minutes

                # Create the GroqAPI instance
                GroqAPI(model_name="test-model", api_key="test-key", config=config)

                # Verify that httpx.AsyncClient was called with the correct timeout
                mock_client.assert_called_once()
                call_args = mock_client.call_args

                # Check that timeout was set correctly
                assert "timeout" in call_args[1]
                timeout_obj = call_args[1]["timeout"]
                # httpx.Timeout object has the timeout value in read/connect/write attributes
                assert timeout_obj.read == 300

    def test_no_timeout_when_not_in_config(self):
        """Test that no timeout is set when not specified in config."""
        # Mock the httpx.AsyncClient to avoid actual HTTP calls
        with patch("httpx.AsyncClient") as mock_client:
            with patch("openbench.model._providers.groq.AsyncGroq"):
                # Create a config without timeout
                config = GenerateConfig()  # No timeout set

                # Create the GroqAPI instance
                GroqAPI(model_name="test-model", api_key="test-key", config=config)

                # Verify that httpx.AsyncClient was called without timeout
                mock_client.assert_called_once()
                call_args = mock_client.call_args

                # Should not have timeout parameter when not specified
                assert "timeout" not in call_args[1]

    def test_no_timeout_when_none_in_config(self):
        """Test that no timeout is set when config timeout is None."""
        # Mock the httpx.AsyncClient to avoid actual HTTP calls
        with patch("httpx.AsyncClient") as mock_client:
            with patch("openbench.model._providers.groq.AsyncGroq"):
                # Create a config with explicit None timeout
                config = GenerateConfig(timeout=None)

                # Create the GroqAPI instance
                GroqAPI(model_name="test-model", api_key="test-key", config=config)

                # Verify that httpx.AsyncClient was called without timeout
                mock_client.assert_called_once()
                call_args = mock_client.call_args

                # Should not have timeout parameter
                assert "timeout" not in call_args[1]

    def test_httpx_timeout_object_creation(self):
        """Test that httpx.Timeout object is created correctly."""
        # Mock the httpx.AsyncClient to avoid actual HTTP calls
        with patch("httpx.AsyncClient") as mock_client:
            with patch("openbench.model._providers.groq.AsyncGroq"):
                with patch("httpx.Timeout") as mock_timeout:
                    # Create a config with timeout
                    config = GenerateConfig(timeout=180)

                    # Create the GroqAPI instance
                    GroqAPI(model_name="test-model", api_key="test-key", config=config)

                    # Verify that httpx.Timeout was called with the correct timeout
                    mock_timeout.assert_called_once_with(timeout=180)

                    # Verify that httpx.AsyncClient was called with the timeout object
                    mock_client.assert_called_once()
                    call_args = mock_client.call_args
                    assert "timeout" in call_args[1]
                    assert call_args[1]["timeout"] == mock_timeout.return_value

    def test_timeout_zero_value(self):
        """Test that timeout=0 is handled correctly."""
        # Mock the httpx.AsyncClient to avoid actual HTTP calls
        with patch("httpx.AsyncClient") as mock_client:
            with patch("openbench.model._providers.groq.AsyncGroq"):
                # Create a config with timeout=0
                config = GenerateConfig(timeout=0)

                # Create the GroqAPI instance
                GroqAPI(model_name="test-model", api_key="test-key", config=config)

                # Verify that httpx.AsyncClient was called with timeout=0
                mock_client.assert_called_once()
                call_args = mock_client.call_args

                # Should have timeout parameter even with 0 value
                assert "timeout" in call_args[1]
                timeout_obj = call_args[1]["timeout"]
                assert timeout_obj.read == 0

    def test_multiple_client_instances(self):
        """Test that multiple instances work correctly with different timeouts."""
        # Mock the httpx.AsyncClient to avoid actual HTTP calls
        with patch("httpx.AsyncClient") as mock_client:
            with patch("openbench.model._providers.groq.AsyncGroq"):
                # Create two instances with different timeouts
                config1 = GenerateConfig(timeout=100)
                config2 = GenerateConfig(timeout=200)

                GroqAPI(model_name="test-model-1", api_key="test-key", config=config1)

                GroqAPI(model_name="test-model-2", api_key="test-key", config=config2)

                # Verify that httpx.AsyncClient was called twice
                assert mock_client.call_count == 2

                # Check first call
                call1_args = mock_client.call_args_list[0]
                assert "timeout" in call1_args[1]
                assert call1_args[1]["timeout"].read == 100

                # Check second call
                call2_args = mock_client.call_args_list[1]
                assert "timeout" in call2_args[1]
                assert call2_args[1]["timeout"].read == 200
