"""
Generic chat completions format provider for Chat Completions API.
"""

import os
from typing import Any, cast

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion
from typing_extensions import override

from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.model._openai import (
    OpenAIAsyncHttpxClient,
    chat_choices_from_openai,
    messages_to_openai,
    model_output_from_openai,
    openai_chat_tool_choice,
    openai_chat_tools,
    openai_completion_params,
    openai_media_filter,
    openai_should_retry,
)
from inspect_ai.model._providers.util.hooks import HttpxHooks
from inspect_ai.tool import ToolChoice, ToolInfo


class ChatCompletionsFormatAPI(ModelAPI):
    """
    Provider for Chat Completions API format.
    """

    def __init__(
        self,
        model_name: str,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        """
        Initialize chat completions format provider.

        Args:
            model_name: Full model string (format: chat-completions/provider/model)
            config: Generation configuration
            **model_args: Additional model arguments
        """
        # Clean format prefix from model name
        model_name_clean = model_name.replace("chat-completions/", "", 1)

        # Extract provider name from model string (pattern: provider/model)
        parts = model_name_clean.split("/")
        if len(parts) == 1:
            raise ValueError(
                "chat-completions model names must include a provider prefix "
                "(e.g. 'chat-completions/provider/model'). "
                "The provider name is used to lookup environment variables PROVIDER_API_KEY and and PROVIDER_BASE_URL."
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
        Generate a response using the Chat Completions API.

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
        request: dict[str, Any] = {}
        response: dict[str, Any] = {}

        def model_call() -> ModelCall:
            return ModelCall.create(
                request=request,
                response=response,
                filter=openai_media_filter,
                time=self._http_hooks.end_request(request_id),
            )

        try:
            # Get completion params
            completion_params = openai_completion_params(
                model=self.actual_model_name,
                config=config,
                tools=len(tools) > 0,
            )

            # Prepare request
            have_tools = len(tools) > 0
            request = dict(
                messages=await messages_to_openai(input),
                tools=openai_chat_tools(tools) if have_tools else NOT_GIVEN,
                tool_choice=openai_chat_tool_choice(tool_choice)
                if have_tools
                else NOT_GIVEN,
                extra_headers={HttpxHooks.REQUEST_ID_HEADER: request_id},
                **completion_params,
            )

            # Generate completion using OpenAI SDK
            completion = cast(
                ChatCompletion, await self.client.chat.completions.create(**request)
            )
            response = completion.model_dump()

            # Parse choices
            choices = chat_choices_from_openai(completion, tools)

            # Return output
            return model_output_from_openai(completion, choices), model_call()

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
