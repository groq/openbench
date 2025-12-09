"""
Groq Responses API provider.

Uses Groq's Responses API (compatible with OpenAI's Responses API) instead of the
Chat Completions API. This enables access to features like MCP, browser_search,
code_execution, and other built-in tools.

Usage: groq-responses/openai/gpt-oss-20b
"""

import json
import os
from copy import copy
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from openai._exceptions import APIStatusError, APITimeoutError
from pydantic import JsonValue
from typing_extensions import override

from inspect_ai._util.constants import (
    BASE_64_DATA_REMOVED,
    DEFAULT_MAX_TOKENS,
)
from inspect_ai._util.content import Content, ContentReasoning, ContentText
from inspect_ai._util.http import is_retryable_http_status
from inspect_ai._util.images import file_as_data_uri
from inspect_ai._util.url import is_http_url
from inspect_ai.tool import ToolCall, ToolChoice, ToolFunction, ToolInfo

from inspect_ai.model._call_tools import parse_tool_call
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    ModelOutput,
    ModelUsage,
    StopReason,
)
from inspect_ai.model._providers.util import (
    environment_prerequisite_error,
    model_base_url,
)
from inspect_ai.model._providers.util.hooks import HttpxHooks

GROQ_API_KEY = "GROQ_API_KEY"


class GroqResponsesAPI(ModelAPI):
    """Groq provider using the Responses API."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=[GROQ_API_KEY],
            config=config,
        )

        if not self.api_key:
            self.api_key = os.environ.get(GROQ_API_KEY)
        if not self.api_key:
            raise environment_prerequisite_error("Groq Responses", GROQ_API_KEY)

        # Extract stream parameter (defaults to True for streaming)
        self.stream = model_args.pop("stream", True)

        # Create httpx client with proper timeout configuration
        timeout_seconds = getattr(config, "timeout", None)
        if timeout_seconds is not None:
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=None),
                timeout=httpx.Timeout(timeout=timeout_seconds),
            )
        else:
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=None),
            )

        # Use OpenAI client with Groq's base URL for Responses API
        resolved_base_url = model_base_url(
            base_url, "GROQ_BASE_URL"
        ) or "https://api.groq.com/openai/v1"

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=resolved_base_url,
            http_client=http_client,
            **model_args,
        )

        # Create time tracker
        self._http_hooks = HttpxHooks(self.client._client)

    @override
    async def aclose(self) -> None:
        await self.client.close()

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput | Exception, ModelCall]:
        request_id = self._http_hooks.start_request()

        request: dict[str, Any] = {}
        response: dict[str, Any] = {}

        def model_call() -> ModelCall:
            return ModelCall.create(
                request=request,
                response=response,
                filter=model_call_filter,
                time=self._http_hooks.end_request(request_id),
            )

        # Convert messages to Responses API format
        messages = await self._convert_messages_for_responses_api(input)

        params = self._build_params(config)

        # Add tools if provided
        if tools:
            params["tools"] = self._build_tools(tools)
            if tool_choice:
                params["tool_choice"] = self._build_tool_choice(tool_choice)

        request = dict(
            input=messages,
            model=self.model_name,
            extra_headers={
                HttpxHooks.REQUEST_ID_HEADER: request_id,
                "User-Agent": "openbench",
            },
            **params,
        )

        try:
            if self.stream:
                result = await self._handle_streaming_response(request, tools)
            else:
                result = await self.client.responses.create(**request)

            response = (
                result.model_dump()
                if hasattr(result, "model_dump")
                else self._result_to_dict(result)
            )

            # Extract output from Responses API format
            output = self._parse_response(result, tools)
            return output, model_call()

        except APIStatusError as ex:
            return self._handle_bad_request(ex), model_call()

    def _build_params(self, config: GenerateConfig) -> Dict[str, Any]:
        """Build parameters for the Responses API."""
        params: dict[str, Any] = {}

        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.max_tokens is not None:
            params["max_output_tokens"] = config.max_tokens
        if config.top_p is not None:
            params["top_p"] = config.top_p

        # Responses API uses 'reasoning' parameter for chain-of-thought
        if config.reasoning_effort is not None:
            params["reasoning"] = {"effort": config.reasoning_effort}

        # Structured outputs / response schema
        if config.response_schema is not None:
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": config.response_schema.name,
                    "schema": config.response_schema.json_schema.model_dump(
                        exclude_none=True
                    ),
                    "strict": config.response_schema.strict,
                }
            }

        if self.stream:
            params["stream"] = True

        return params

    def _build_tools(self, tools: List[ToolInfo]) -> List[Dict[str, Any]]:
        """Convert ToolInfo to Responses API tool format.

        Responses API expects: {"type": "function", "name": "...", "description": "...", "parameters": {...}}
        Not the nested format: {"type": "function", "function": {...}}
        """
        result = []
        for tool in tools:
            tool_dict = tool.model_dump(exclude_none=True)
            result.append(
                {
                    "type": "function",
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": tool_dict.get("parameters", {}),
                }
            )
        return result

    def _build_tool_choice(self, tool_choice: ToolChoice) -> str | Dict[str, Any]:
        """Convert ToolChoice to Responses API format."""
        if isinstance(tool_choice, ToolFunction):
            return {"type": "function", "function": {"name": tool_choice.name}}
        elif tool_choice == "any":
            return "auto"
        else:
            return tool_choice

    async def _convert_messages_for_responses_api(
        self, messages: list[ChatMessage]
    ) -> list[Dict[str, Any]]:
        """Convert ChatMessages to Responses API input format."""
        result: list[Dict[str, Any]] = []

        for message in messages:
            if isinstance(message, ChatMessageSystem):
                result.append({"role": "system", "content": message.text})
            elif isinstance(message, ChatMessageUser):
                user_content = await self._convert_user_content(message.content)
                result.append({"role": "user", "content": user_content})
            elif isinstance(message, ChatMessageAssistant):
                # If assistant has tool calls, add them as function_call items
                # (Responses API uses function_call items, not assistant messages with tool_calls)
                if message.tool_calls:
                    for call in message.tool_calls:
                        result.append(
                            {
                                "type": "function_call",
                                "call_id": call.id,
                                "name": call.function,
                                "arguments": json.dumps(call.arguments),
                            }
                        )
                elif message.text:
                    # Only add assistant message if it has text content
                    result.append({"role": "assistant", "content": message.text})
            elif isinstance(message, ChatMessageTool):
                # Responses API uses function_call_output type instead of tool role
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.tool_call_id),
                        "output": message.text,
                    }
                )

        return result

    async def _convert_user_content(
        self, content: str | list[Content]
    ) -> str | list[Dict[str, Any]]:
        """Convert user content to Responses API format."""
        if isinstance(content, str):
            return content

        parts = []
        for item in content:
            if item.type == "text":
                parts.append({"type": "input_text", "text": item.text})
            elif item.type == "image":
                image_url = item.image
                if not is_http_url(image_url):
                    image_url = await file_as_data_uri(image_url)
                parts.append(
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": item.detail,
                    }
                )
        return parts

    def _parse_response(
        self, result: Any, tools: list[ToolInfo]
    ) -> ModelOutput:
        """Parse Responses API result into ModelOutput."""
        # Extract content from output items
        text_content = ""
        reasoning_content = ""
        tool_calls: List[ToolCall] = []

        # Responses API returns output as a list of items
        output_items = getattr(result, "output", []) or []

        for item in output_items:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                # Message contains content list
                message_content = getattr(item, "content", []) or []
                for content_item in message_content:
                    content_type = getattr(content_item, "type", None)
                    if content_type == "output_text":
                        text_content += getattr(content_item, "text", "")
                    elif content_type == "refusal":
                        text_content += getattr(content_item, "refusal", "")

            elif item_type == "reasoning":
                # Reasoning trace
                summary = getattr(item, "summary", []) or []
                for s in summary:
                    if hasattr(s, "text"):
                        reasoning_content += s.text

            elif item_type == "function_call":
                # Tool/function call
                call_id = getattr(item, "call_id", "")
                name = getattr(item, "name", "")
                arguments = getattr(item, "arguments", "{}")
                tool_calls.append(parse_tool_call(call_id, name, arguments, tools))

        # Build content
        if reasoning_content:
            content: str | list[Content] = [
                ContentReasoning(reasoning=reasoning_content),
                ContentText(text=text_content),
            ]
        else:
            content = text_content

        # Determine stop reason
        stop_reason: StopReason = "stop"
        if tool_calls:
            stop_reason = "tool_calls"

        # Extract usage
        usage = None
        if hasattr(result, "usage") and result.usage:
            usage = ModelUsage(
                input_tokens=getattr(result.usage, "input_tokens", 0),
                output_tokens=getattr(result.usage, "output_tokens", 0),
                total_tokens=getattr(result.usage, "total_tokens", 0),
            )

        # Build message
        assistant_message = ChatMessageAssistant(
            content=content,
            model=self.model_name,
            source="generate",
            tool_calls=tool_calls if tool_calls else None,
        )

        return ModelOutput(
            model=getattr(result, "model", self.model_name),
            choices=[
                ChatCompletionChoice(
                    message=assistant_message,
                    stop_reason=stop_reason,
                )
            ],
            usage=usage,
            metadata={
                "id": getattr(result, "id", ""),
            },
        )

    async def _handle_streaming_response(
        self, request: Dict[str, Any], tools: list[ToolInfo]
    ) -> Any:
        """Handle streaming response from Responses API."""
        stream = await self.client.responses.create(**request)

        # Accumulate streamed content
        accumulated_text = ""
        accumulated_reasoning = ""
        accumulated_tool_calls: Dict[str, Dict[str, Any]] = {}
        usage = None
        response_id = ""
        model_name = ""

        async for event in stream:
            event_type = getattr(event, "type", "")

            if event_type == "response.created":
                response = getattr(event, "response", None)
                if response:
                    response_id = getattr(response, "id", "")
                    model_name = getattr(response, "model", self.model_name)

            elif event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                accumulated_text += delta

            elif event_type == "response.reasoning.delta":
                delta = getattr(event, "delta", "")
                accumulated_reasoning += delta

            elif event_type in ("response.output_item.added", "response.output_item.done"):
                # Function call item - capture the name and potentially final arguments
                item = getattr(event, "item", None)
                if item and getattr(item, "type", "") == "function_call":
                    # Try multiple ID field names
                    call_id = (
                        getattr(item, "call_id", "")
                        or getattr(item, "id", "")
                        or getattr(item, "item_id", "")
                    )
                    name = getattr(item, "name", "")
                    arguments = getattr(item, "arguments", "")
                    if call_id:
                        if call_id not in accumulated_tool_calls:
                            accumulated_tool_calls[call_id] = {
                                "id": call_id,
                                "name": name,
                                "arguments": arguments or "",
                            }
                        else:
                            # Update with any new data
                            if name:
                                accumulated_tool_calls[call_id]["name"] = name
                            if arguments:
                                accumulated_tool_calls[call_id]["arguments"] = arguments

            elif event_type == "response.function_call_arguments.delta":
                # Accumulate function call arguments
                # Groq uses item_id instead of call_id
                call_id = (
                    getattr(event, "call_id", "")
                    or getattr(event, "item_id", "")
                )
                if call_id and call_id not in accumulated_tool_calls:
                    accumulated_tool_calls[call_id] = {
                        "id": call_id,
                        "name": getattr(event, "name", "") or "",
                        "arguments": "",
                    }
                if call_id:
                    accumulated_tool_calls[call_id]["arguments"] += getattr(
                        event, "delta", ""
                    )

            elif event_type == "response.function_call_arguments.done":
                # Groq uses item_id instead of call_id
                call_id = (
                    getattr(event, "call_id", "")
                    or getattr(event, "item_id", "")
                )
                if call_id:
                    if call_id not in accumulated_tool_calls:
                        accumulated_tool_calls[call_id] = {
                            "id": call_id,
                            "name": "",
                            "arguments": "",
                        }
                    name = getattr(event, "name", "")
                    if name:
                        accumulated_tool_calls[call_id]["name"] = name
                    accumulated_tool_calls[call_id]["arguments"] = getattr(
                        event, "arguments", ""
                    )

            elif event_type == "response.completed":
                response = getattr(event, "response", None)
                if response and hasattr(response, "usage"):
                    usage = response.usage

        # Build mock result object - only include tool calls with valid names
        tool_call_list = []
        for tc in accumulated_tool_calls.values():
            if tc["name"]:  # Skip tool calls with empty names
                tool_call_list.append(
                    parse_tool_call(tc["id"], tc["name"], tc["arguments"], tools)
                )

        # Create a result-like object
        result = type(
            "MockResponse",
            (),
            {
                "id": response_id,
                "model": model_name,
                "output": [
                    type(
                        "MessageItem",
                        (),
                        {
                            "type": "message",
                            "content": [
                                type(
                                    "TextContent",
                                    (),
                                    {"type": "output_text", "text": accumulated_text},
                                )()
                            ],
                        },
                    )()
                ]
                + (
                    [
                        type(
                            "ReasoningItem",
                            (),
                            {
                                "type": "reasoning",
                                "summary": [
                                    type("Summary", (), {"text": accumulated_reasoning})()
                                ],
                            },
                        )()
                    ]
                    if accumulated_reasoning
                    else []
                )
                + [
                    type(
                        "FunctionCallItem",
                        (),
                        {
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    )()
                    for tc in accumulated_tool_calls.values()
                ],
                "usage": usage,
                "model_dump": lambda self: {
                    "id": response_id,
                    "model": model_name,
                    "output": [],
                    "usage": usage.model_dump() if usage else None,
                },
            },
        )()

        return result

    def _result_to_dict(self, result: Any) -> Dict[str, Any]:
        """Convert result to dictionary for logging."""
        return {
            "id": getattr(result, "id", ""),
            "model": getattr(result, "model", ""),
            "output": str(getattr(result, "output", [])),
        }

    def _handle_bad_request(self, ex: APIStatusError) -> ModelOutput | Exception:
        """Handle API errors."""
        if ex.status_code == 400:
            content = ex.message
            if isinstance(ex.body, dict) and isinstance(ex.body.get("error", None), dict):
                error = ex.body.get("error", {})
                content = str(error.get("message", content))
                code = error.get("code", "")

                if code == "context_length_exceeded":
                    return ModelOutput.from_content(
                        model=self.model_name,
                        content=content,
                        stop_reason="model_length",
                    )

        return ex

    @override
    def should_retry(self, ex: Exception) -> bool:
        if isinstance(ex, APIStatusError):
            return is_retryable_http_status(ex.status_code)
        elif isinstance(ex, APITimeoutError):
            return True
        return False

    @override
    def connection_key(self) -> str:
        return str(self.api_key)

    @override
    def collapse_user_messages(self) -> bool:
        return False

    @override
    def collapse_assistant_messages(self) -> bool:
        return False

    @override
    def max_tokens(self) -> Optional[int]:
        return DEFAULT_MAX_TOKENS


def model_call_filter(key: JsonValue | None, value: JsonValue) -> JsonValue:
    if key == "image_url" and isinstance(value, dict):
        value = copy(value)
        value.update(url=BASE_64_DATA_REMOVED)
    return value
