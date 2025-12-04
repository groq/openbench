"""
Generic responses format provider for Responses API.
"""

import logging
import os
from typing import Any, cast

from openai import AsyncOpenAI, BadRequestError
from openai._types import NOT_GIVEN
from openai.types.responses import Response as OpenAIResponse
from typing_extensions import override

from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import ModelOutput, ModelUsage
from inspect_ai.model._openai import (
    OpenAIAsyncHttpxClient,
    openai_media_filter,
    openai_should_retry,
)
from inspect_ai.model._openai_responses import (
    openai_responses_chat_choices,
    openai_responses_inputs,
    openai_responses_tool_choice,
    openai_responses_tools,
)
from inspect_ai.model._providers.util.hooks import HttpxHooks
from inspect_ai.tool import ToolChoice, ToolInfo

# Set up logger for this module
logger = logging.getLogger(__name__)


class _OpenAIAPIAdapter:
    """
    Minimal adapter to enable Inspect AI responses utilities for generic providers.

    The openai_responses_inputs() function requires an OpenAIAPI instance to check
    model type (o-series vs gpt-5) to determine whether to use "developer" or "system"
    role for system messages. For generic providers, we conservatively use "system" role.
    """

    def __init__(self, model_name: str):
        """Initialize adapter with model name."""
        self.model_name = model_name

    def is_o_series(self) -> bool:
        """Check if model is o-series. Generic providers: return False."""
        return False

    def is_gpt_5(self) -> bool:
        """Check if model is gpt-5. Generic providers: return False."""
        return False


class ResponsesFormatAPI(ModelAPI):
    """
    Provider for Responses API format.
    """

    def __init__(
        self,
        model_name: str,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        """
        Initialize responses format provider.

        Args:
            model_name: Full model string (format: responses/provider/model)
            config: Generation configuration
            **model_args: Additional model arguments
        """
        # Clean format prefix from model name
        model_name_clean = model_name.replace("responses/", "", 1)

        # Extract provider name from model string (pattern: provider/model)
        parts = model_name_clean.split("/")
        if len(parts) == 1:
            raise ValueError(
                "responses model names must include a provider prefix "
                "(e.g. 'responses/provider/model'). "
                "The provider name is used to lookup environment variables PROVIDER_API_KEY and PROVIDER_BASE_URL."
            )

        self.provider = parts[0]
        self.actual_model_name = "/".join(parts[1:])  # Handle nested slashes

        # Compute environment variable names from provider
        provider_env_name = self.provider.upper().replace("-", "_")
        api_key_var = f"{provider_env_name}_API_KEY"
        base_url_var = f"{provider_env_name}_BASE_URL"

        # Get API key and base URL from environment variables
        api_key = os.environ.get(api_key_var)

        base_url = os.environ.get(base_url_var)
        if not base_url:
            raise ValueError(
                f"Base URL is required. Set {base_url_var} environment variable."
            )

        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=[api_key_var],
            config=config,
        )

        # Create OpenAI client
        http_client = model_args.pop("http_client", OpenAIAsyncHttpxClient())
        # Remove base_url and api_key from model_args to avoid conflicts
        model_args.pop("base_url", None)
        model_args.pop("api_key", None)

        self.client = AsyncOpenAI(
            api_key=self.api_key or "dummy",  # OpenAI client requires a key
            base_url=self.base_url,
            http_client=http_client,
            **model_args,
        )

        # Create time tracker
        self._http_hooks = HttpxHooks(self.client._client)

    @override
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput | Exception, ModelCall]:
        """
        Generate a response using the OpenAI Responses API.

        Args:
            input: Chat messages
            tools: Available tools
            tool_choice: Tool selection preference
            config: Generation configuration

        Returns:
            Tuple of (ModelOutput or Exception, ModelCall)
        """
        # Allocate request ID
        request_id = self._http_hooks.start_request()

        # Setup request and response for ModelCall
        request_dict: dict[str, Any] = {}
        response_dict: dict[str, Any] = {}

        def model_call() -> ModelCall:
            return ModelCall.create(
                request=request_dict,
                response=response_dict,
                filter=openai_media_filter,
                time=self._http_hooks.end_request(request_id),
            )

        try:
            # Convert messages to responses input format
            openai_api_adapter = _OpenAIAPIAdapter(self.actual_model_name)
            input_items = await openai_responses_inputs(input, openai_api_adapter)  # type: ignore[arg-type]

            # Extract instructions from first system message
            instructions: str | None = None
            filtered_items: list[Any] = []
            for item in input_items:
                # Check if this is a message item with system role
                if (
                    isinstance(item, dict)
                    and item.get("type") == "message"  # type: ignore[misc]
                    and item.get("role") == "system"  # type: ignore[misc]
                    and instructions is None
                ):
                    # Extract content as instructions
                    content: Any = item.get("content", [])  # type: ignore[misc]
                    if content:
                        # Join text content from content array
                        texts = []
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":  # type: ignore[misc]
                                texts.append(c.get("text", ""))  # type: ignore[misc]
                        instructions = "\n".join(texts) if texts else None
                else:
                    filtered_items.append(item)

            # Build completion params
            params: dict[str, Any] = {
                "model": self.actual_model_name,
            }

            # Add supported config parameters
            # Responses API uses max_output_tokens (not max_tokens)
            if config.max_tokens is not None:
                params["max_output_tokens"] = config.max_tokens
            if config.temperature is not None:
                params["temperature"] = config.temperature
            if config.top_p is not None:
                params["top_p"] = config.top_p

            # Unsupported params are silently ignored

            # Convert tools to responses format
            have_tools = len(tools) > 0
            if have_tools:
                tool_params: Any = openai_responses_tools(
                    tools, self.actual_model_name, config
                )
                tool_choice_param: Any = openai_responses_tool_choice(
                    tool_choice, tool_params
                )
            else:
                tool_params = NOT_GIVEN
                tool_choice_param = NOT_GIVEN

            # Prepare request
            request_dict = {
                "input": filtered_items,
                "instructions": instructions if instructions else NOT_GIVEN,
                "tools": tool_params,
                "tool_choice": tool_choice_param,
                "extra_headers": {HttpxHooks.REQUEST_ID_HEADER: request_id},
                **params,
            }

            # Generate response using OpenAI SDK
            try:
                openai_response = cast(
                    OpenAIResponse,
                    await self.client.responses.create(**request_dict),  # type: ignore[arg-type]
                )
            except BadRequestError as e:
                # Check if this is an unsupported parameter error
                error_message = str(e)
                if (
                    "Unsupported parameter" in error_message
                    or "not supported" in error_message
                ):
                    # Retry without optional parameters
                    logger.warning(
                        f"Model {self.actual_model_name} does not support some parameters. "
                        f"Attempting to continue with the model's default values. "
                        f"Original error: {error_message}"
                    )

                    # Build minimal request without optional params
                    minimal_params = {"model": self.actual_model_name}
                    minimal_request_dict: dict[str, Any] = {
                        "input": filtered_items,
                        "instructions": instructions if instructions else NOT_GIVEN,
                        "tools": tool_params,
                        "tool_choice": tool_choice_param,
                        "extra_headers": {HttpxHooks.REQUEST_ID_HEADER: request_id},
                        **minimal_params,
                    }

                    # Retry with minimal params
                    openai_response = cast(
                        OpenAIResponse,
                        await self.client.responses.create(**minimal_request_dict),  # type: ignore[arg-type]
                    )
                else:
                    # Not an unsupported parameter error, re-raise
                    raise

            response_dict = openai_response.model_dump()

            # Parse choices using Inspect AI utility
            choices = openai_responses_chat_choices(
                self.actual_model_name,
                openai_response,
                tools,
            )

            # Extract usage if available
            usage = None
            if openai_response.usage:
                usage = ModelUsage(
                    input_tokens=openai_response.usage.input_tokens,
                    output_tokens=openai_response.usage.output_tokens,
                    total_tokens=openai_response.usage.total_tokens,
                )

            # Build ModelOutput
            return ModelOutput(
                model=openai_response.model,
                choices=choices,
                usage=usage,
            ), model_call()

        except Exception as ex:
            return ex, model_call()

    @override
    def should_retry(self, ex: Exception) -> bool:
        """Determine if a request should be retried."""
        return openai_should_retry(ex)

    @override
    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self.client.close()

    @override
    def connection_key(self) -> str:
        """Get the connection key for connection pooling."""
        return str(self.api_key or self.base_url)

    @override
    def collapse_user_messages(self) -> bool:
        """Whether to collapse consecutive user messages."""
        return False

    @override
    def collapse_assistant_messages(self) -> bool:
        """Whether to collapse consecutive assistant messages."""
        return False
