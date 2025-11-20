"""Unit tests for Groq provider timeout configuration and streaming support."""

from unittest.mock import AsyncMock, MagicMock, patch
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


class TestGroqProviderStreaming:
    """Test Groq provider streaming support."""

    def test_stream_parameter_extraction_true(self):
        """Test that stream=True is properly extracted from model args."""
        with patch("httpx.AsyncClient"):
            with patch("openbench.model._providers.groq.AsyncGroq"):
                config = GenerateConfig()
                model_args = {"stream": True, "other_arg": "value"}
                
                groq_api = GroqAPI(
                    model_name="test-model", 
                    api_key="test-key", 
                    config=config,
                    **model_args
                )
                
                # Stream parameter should be extracted and set
                assert groq_api.stream is True
                
    def test_stream_parameter_extraction_false(self):
        """Test that stream=False is properly extracted from model args."""
        with patch("httpx.AsyncClient"):
            with patch("openbench.model._providers.groq.AsyncGroq"):
                config = GenerateConfig()
                model_args = {"stream": False, "other_arg": "value"}
                
                groq_api = GroqAPI(
                    model_name="test-model", 
                    api_key="test-key", 
                    config=config,
                    **model_args
                )
                
                # Stream parameter should be extracted and set
                assert groq_api.stream is False

    def test_stream_parameter_default_true(self):
        """Test that stream defaults to True when not provided."""
        with patch("httpx.AsyncClient"):
            with patch("openbench.model._providers.groq.AsyncGroq"):
                config = GenerateConfig()
                model_args = {"other_arg": "value"}
                
                groq_api = GroqAPI(
                    model_name="test-model", 
                    api_key="test-key", 
                    config=config,
                    **model_args
                )
                
                # Stream parameter should default to True
                assert groq_api.stream is True

    def test_stream_parameter_removed_from_client_args(self):
        """Test that stream parameter is removed from model_args passed to AsyncGroq."""
        with patch("httpx.AsyncClient"):
            with patch("openbench.model._providers.groq.AsyncGroq") as mock_groq:
                config = GenerateConfig()
                model_args = {"stream": True, "other_arg": "value"}
                
                GroqAPI(
                    model_name="test-model", 
                    api_key="test-key", 
                    config=config,
                    **model_args
                )
                
                # Verify AsyncGroq was called without the stream parameter
                mock_groq.assert_called_once()
                call_kwargs = mock_groq.call_args[1]
                assert "stream" not in call_kwargs
                assert call_kwargs["other_arg"] == "value"

    def test_completion_params_includes_stream_when_enabled(self):
        """Test that completion_params includes stream=True when streaming is enabled."""
        with patch("httpx.AsyncClient"):
            with patch("openbench.model._providers.groq.AsyncGroq"):
                config = GenerateConfig()
                model_args = {"stream": True}
                
                groq_api = GroqAPI(
                    model_name="test-model", 
                    api_key="test-key", 
                    config=config,
                    **model_args
                )
                
                params = groq_api.completion_params(config)
                assert params["stream"] is True

    def test_completion_params_includes_stream_by_default(self):
        """Test that completion_params includes stream=True by default."""
        with patch("httpx.AsyncClient"):
            with patch("openbench.model._providers.groq.AsyncGroq"):
                config = GenerateConfig()
                model_args = {}  # No stream parameter specified
                
                groq_api = GroqAPI(
                    model_name="test-model", 
                    api_key="test-key", 
                    config=config,
                    **model_args
                )
                
                params = groq_api.completion_params(config)
                assert params["stream"] is True

    async def test_handle_streaming_response_content_accumulation(self):
        """Test that streaming response properly accumulates content chunks."""
        with patch("httpx.AsyncClient"):
            with patch("openbench.model._providers.groq.AsyncGroq"):
                config = GenerateConfig()
                groq_api = GroqAPI(
                    model_name="test-model", 
                    api_key="test-key", 
                    config=config,
                    stream=True
                )
                
                # Create mock stream chunks
                chunk1 = MagicMock()
                chunk1.choices = [MagicMock()]
                chunk1.choices[0].delta.content = "Hello "
                chunk1.choices[0].delta.reasoning = None
                chunk1.choices[0].delta.tool_calls = None
                chunk1.choices[0].delta.executed_tools = None
                chunk1.choices[0].finish_reason = None
                chunk1.id = "test-id"
                chunk1.model = "test-model"
                chunk1.system_fingerprint = "test-fingerprint"
                chunk1.created = 1234567890
                
                chunk2 = MagicMock()
                chunk2.choices = [MagicMock()]
                chunk2.choices[0].delta.content = "world!"
                chunk2.choices[0].delta.reasoning = None
                chunk2.choices[0].delta.tool_calls = None
                chunk2.choices[0].delta.executed_tools = None
                chunk2.choices[0].finish_reason = "stop"
                chunk2.id = "test-id"
                
                # Mock async iterator
                mock_stream = AsyncMock()
                mock_stream.__aiter__.return_value = [chunk1, chunk2]
                
                result = await groq_api._handle_streaming_response(mock_stream, [])
                
                # Verify content was accumulated
                assert result.choices[0].message.content == "Hello world!"
                assert result.choices[0].finish_reason == "stop"
                assert result.id == "test-id"
                assert result.model == "test-model"

    async def test_handle_streaming_response_reasoning_accumulation(self):
        """Test that streaming response properly accumulates reasoning chunks."""
        with patch("httpx.AsyncClient"):
            with patch("openbench.model._providers.groq.AsyncGroq"):
                config = GenerateConfig()
                groq_api = GroqAPI(
                    model_name="test-model", 
                    api_key="test-key", 
                    config=config,
                    stream=True
                )
                
                # Create mock stream chunks with reasoning
                chunk1 = MagicMock()
                chunk1.choices = [MagicMock()]
                chunk1.choices[0].delta.content = None
                chunk1.choices[0].delta.reasoning = "First reasoning "
                chunk1.choices[0].delta.tool_calls = None
                chunk1.choices[0].delta.executed_tools = None
                chunk1.choices[0].finish_reason = None
                chunk1.id = "test-id"
                chunk1.model = "test-model"
                
                chunk2 = MagicMock()
                chunk2.choices = [MagicMock()]
                chunk2.choices[0].delta.content = None
                chunk2.choices[0].delta.reasoning = "second reasoning"
                chunk2.choices[0].delta.tool_calls = None
                chunk2.choices[0].delta.executed_tools = None
                chunk2.choices[0].finish_reason = "stop"
                
                mock_stream = AsyncMock()
                mock_stream.__aiter__.return_value = [chunk1, chunk2]
                
                result = await groq_api._handle_streaming_response(mock_stream, [])
                
                # Verify reasoning was accumulated
                assert result.choices[0].message.reasoning == "First reasoning second reasoning"

    async def test_handle_streaming_response_tool_calls_accumulation(self):
        """Test that streaming response properly accumulates tool calls."""
        with patch("httpx.AsyncClient"):
            with patch("openbench.model._providers.groq.AsyncGroq"):
                config = GenerateConfig()
                groq_api = GroqAPI(
                    model_name="test-model", 
                    api_key="test-key", 
                    config=config,
                    stream=True
                )
                
                # Create mock stream chunks with tool calls
                tool_call1 = MagicMock()
                tool_call1.index = 0
                tool_call1.id = "call-123"
                tool_call1.type = "function"
                tool_call1.function.name = "test_func"
                tool_call1.function.arguments = '{"param":'
                
                tool_call2 = MagicMock()
                tool_call2.index = 0
                tool_call2.id = None
                tool_call2.type = None
                tool_call2.function.name = None
                tool_call2.function.arguments = ' "value"}'
                
                chunk1 = MagicMock()
                chunk1.choices = [MagicMock()]
                chunk1.choices[0].delta.content = None
                chunk1.choices[0].delta.reasoning = None
                chunk1.choices[0].delta.tool_calls = [tool_call1]
                chunk1.choices[0].delta.executed_tools = None
                chunk1.choices[0].finish_reason = None
                chunk1.id = "test-id"
                chunk1.model = "test-model"
                
                chunk2 = MagicMock()
                chunk2.choices = [MagicMock()]
                chunk2.choices[0].delta.content = None
                chunk2.choices[0].delta.reasoning = None
                chunk2.choices[0].delta.tool_calls = [tool_call2]
                chunk2.choices[0].delta.executed_tools = None
                chunk2.choices[0].finish_reason = "tool_calls"
                
                mock_stream = AsyncMock()
                mock_stream.__aiter__.return_value = [chunk1, chunk2]
                
                result = await groq_api._handle_streaming_response(mock_stream, [])
                
                # Verify tool call was accumulated
                assert len(result.choices[0].message.tool_calls) == 1
                tool_call = result.choices[0].message.tool_calls[0]
                assert tool_call.id == "call-123"
                assert tool_call.type == "function"
                assert tool_call.function.name == "test_func"
                assert tool_call.function.arguments == '{"param": "value"}'

    async def test_handle_streaming_response_empty_stream(self):
        """Test that streaming response handles empty stream gracefully."""
        with patch("httpx.AsyncClient"):
            with patch("openbench.model._providers.groq.AsyncGroq"):
                config = GenerateConfig()
                groq_api = GroqAPI(
                    model_name="test-model", 
                    api_key="test-key", 
                    config=config,
                    stream=True
                )
                
                # Empty stream
                mock_stream = AsyncMock()
                mock_stream.__aiter__.return_value = []
                
                result = await groq_api._handle_streaming_response(mock_stream, [])
                
                # Should return valid but empty response
                assert result.choices[0].message.content == ""
                assert result.choices[0].message.reasoning is None
                assert result.choices[0].message.tool_calls is None

    async def test_handle_streaming_response_usage_extraction(self):
        """Test that streaming response extracts usage from x_groq field."""
        with patch("httpx.AsyncClient"):
            with patch("openbench.model._providers.groq.AsyncGroq"):
                config = GenerateConfig()
                groq_api = GroqAPI(
                    model_name="test-model", 
                    api_key="test-key", 
                    config=config,
                    stream=True
                )
                
                # Create chunk with x_groq usage
                chunk = MagicMock()
                chunk.choices = []
                chunk.x_groq = MagicMock()
                chunk.x_groq.usage = MagicMock()
                chunk.x_groq.usage.prompt_tokens = 10
                chunk.x_groq.usage.completion_tokens = 20
                chunk.x_groq.usage.total_tokens = 30
                
                mock_stream = AsyncMock()
                mock_stream.__aiter__.return_value = [chunk]
                
                result = await groq_api._handle_streaming_response(mock_stream, [])
                
                # Verify usage was extracted
                assert result.usage == chunk.x_groq.usage
