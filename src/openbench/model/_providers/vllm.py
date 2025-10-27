# Adapted from https://github.com/UKGovernmentBEIS/inspect_evals

import functools
import logging

from openai import APIStatusError
from typing_extensions import override

from inspect_ai._util.content import (
    Content,
    ContentImage,
    ContentReasoning,
    ContentText,
)
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.tool._tool_choice import ToolChoice
from inspect_ai.tool._tool_info import ToolInfo

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

# Set up logger for this module
logger = logging.getLogger(__name__)


class VLLMAPI(OpenAICompatibleAPI):
    """
    Provider for using vLLM models via an existing vLLM server endpoint.

    This provider connects to an existing vLLM server and uses it for inference.

    Args:
        model_name (str): Name of the model to use, e.g. "Devstral-Small-2505"
        base_url (str | None): Base URL of the vLLM server. If not provided, will use localhost:8000.
        port (int | None): Port of the vLLM server. If provided, will construct base_url as http://localhost:{port}/v1.
        api_key (str | None): API key for the vLLM server. If not provided, will use "openbench" as default.
        config (GenerateConfig): Configuration for generation. Defaults to GenerateConfig().
        is_mistral (bool): Whether the model is a Mistral model. If True, it will handle folding user messages into tool messages as Mistral does not support a user message immediately after a tool message. Defaults to False.

    Environment variables:
        VLLM_BASE_URL: Base URL for an existing vLLM server
        VLLM_API_KEY: API key for the vLLM server
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        is_mistral: bool = False,
    ) -> None:
        # Validate inputs
        if base_url and port:
            raise ValueError("base_url and port cannot both be provided.")

        # Set default base_url if not provided
        if not base_url:
            if port:
                base_url = f"http://localhost:{port}/v1"
            else:
                base_url = "http://localhost:8000/v1"

        # Initialize instance variables
        self.is_mistral = is_mistral

        # Initialize with existing server
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="vLLM",
            service_base_url=base_url,
        )
        logger.info(f"Connected to vLLM server at {self.base_url}")

    @override
    def collapse_user_messages(self) -> bool:
        return True

    @override
    def collapse_assistant_messages(self) -> bool:
        return True

    async def aclose(self) -> None:
        """Close the client."""
        await super().aclose()

    def close(self) -> None:
        """Close the client."""
        # No server cleanup needed since we don't manage the server
        pass

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        # check if last message is an assistant message, in this case we want to
        # continue the final message instead of generating a new one
        if input[-1].role == "assistant":
            # Create a copy of the config to avoid modifying the original
            config = config.model_copy()

            # Set these parameters in extra_body
            if config.extra_body is None:
                config.extra_body = {}

            # Only set these values if they're not already present in extra_body
            if (
                "add_generation_prompt" not in config.extra_body
                and "continue_final_message" not in config.extra_body
            ):
                config.extra_body["add_generation_prompt"] = False
                config.extra_body["continue_final_message"] = True
        # if model is mistral, we need to fold user messages into tool messages, as mistral does not support a user message immediately after a tool message
        if self.is_mistral:
            input = functools.reduce(mistral_message_reducer, input, [])
        return await super().generate(input, tools, tool_choice, config)

    @override
    def handle_bad_request(self, ex: APIStatusError) -> ModelOutput | Exception:
        if ex.status_code == 400:
            # Extract message safely
            if isinstance(ex.body, dict) and "message" in ex.body:
                content = str(ex.body.get("message"))
            else:
                content = ex.message

            if (
                "maximum context length" in content
                or "max_tokens must be at least 1" in content
            ):
                return ModelOutput.from_content(
                    self.model_name, content=content, stop_reason="model_length"
                )
        return ex


def mistral_message_reducer(
    messages: list[ChatMessage],
    message: ChatMessage,
) -> list[ChatMessage]:
    """Fold any user messages found immediately after tool messages into the last tool message."""
    if (
        len(messages) > 0
        and isinstance(messages[-1], ChatMessageTool)
        and isinstance(message, ChatMessageUser)
    ):
        messages[-1] = fold_user_message_into_tool_message(messages[-1], message)
    else:
        messages.append(message)

    return messages


def fold_user_message_into_tool_message(
    tool_message: ChatMessageTool,
    user_message: ChatMessageUser,
) -> ChatMessageTool:
    def convert_content_items_to_string(list_content: list[Content]) -> str:
        if not all(
            isinstance(item, (ContentText | ContentReasoning | ContentImage))
            for item in list_content
        ):
            raise TypeError("Expected all items to be ContentText or ContentReasoning")

        parts = []
        for item in list_content:
            if isinstance(item, ContentText):
                parts.append(item.text)
            elif isinstance(item, ContentReasoning):
                parts.append(item.reasoning)
            elif isinstance(item, ContentImage):
                parts.append(f"[Image: {item.image}]")
            else:
                raise ValueError("Unexpected content item type")
        return "".join(parts)

    def normalise_content(
        content: str | list[Content] | None,
    ) -> str | None:
        return (
            None
            if content is None
            else convert_content_items_to_string(content)
            if isinstance(content, list)
            else content
        )

    tool_content = normalise_content(tool_message.content)
    user_content = normalise_content(user_message.content)

    return ChatMessageTool(
        content=(tool_content or "") + (user_content or ""),
        tool_call_id=tool_message.tool_call_id,
    )
