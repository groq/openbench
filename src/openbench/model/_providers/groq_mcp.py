"""
Groq MCP Provider - extends GroqAPI to support remote MCP tools.

This provider intercepts MCP tool definitions from extra_body and injects
them into the tools array for Groq's native remote MCP support.
"""

from typing import Any, Optional

from typing_extensions import override

from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import ModelOutput, ModelUsage

from .groq import GroqAPI


class GroqMCPAPI(GroqAPI):
    """Groq API with remote MCP tool support.

    This provider extends GroqAPI to support passing MCP server definitions
    through extra_body. When mcp_tools are found in extra_body, they are
    injected into the request's tools array for Groq's native MCP handling.
    """

    @override
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput | Exception, ModelCall]:
        # Check for MCP tools in extra_body
        mcp_tools: list[dict[str, Any]] = []
        mcp_tool_choice: Optional[str] = None

        if config.extra_body:
            mcp_tools = config.extra_body.pop("mcp_tools", [])
            mcp_tool_choice = config.extra_body.pop("mcp_tool_choice", None)

        # If no MCP tools, use standard generation
        if not mcp_tools:
            return await super().generate(input, tools, tool_choice, config)

        # Otherwise, we need to inject MCP tools directly into the request
        # We'll call the Groq client directly with MCP tools
        from inspect_ai.model._providers.util.hooks import HttpxHooks

        from .groq import as_groq_chat_messages, model_call_filter

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

        messages = await as_groq_chat_messages(input)
        params = self.completion_params(config)

        # Add MCP tools directly to params
        params["tools"] = mcp_tools
        params["tool_choice"] = mcp_tool_choice or "auto"

        request = dict(
            messages=messages,
            model=self.model_name,
            extra_headers={
                HttpxHooks.REQUEST_ID_HEADER: request_id,
                "User-Agent": "openbench",
            },
            **params,
        )

        try:
            if self.stream:
                stream = await self.client.chat.completions.create(**request)
                # For MCP, we don't have local tools to validate against
                completion = await self._handle_streaming_response(stream, [])
            else:
                completion = await self.client.chat.completions.create(**request)

            response = completion.model_dump()

            # Extract metadata
            metadata: dict[str, Any] = {
                "id": completion.id,
                "system_fingerprint": completion.system_fingerprint,
                "created": completion.created,
            }
            if completion.usage:
                metadata = metadata | {
                    "queue_time": completion.usage.queue_time,
                    "prompt_time": completion.usage.prompt_time,
                    "completion_time": completion.usage.completion_time,
                    "total_time": completion.usage.total_time,
                }

            # Build choices - no local tools to validate against
            choices = self._chat_choices_from_response(completion, [])

            usage = None
            if completion.usage:
                usage = ModelUsage(
                    input_tokens=completion.usage.prompt_tokens,
                    output_tokens=completion.usage.completion_tokens,
                    total_tokens=completion.usage.total_tokens,
                )

            return (
                ModelOutput(
                    model=self.model_name,
                    choices=choices,
                    usage=usage,
                    metadata=metadata,
                ),
                model_call(),
            )

        except Exception as e:
            return e, model_call()
