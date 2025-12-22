"""
Anthropic remote MCP handler.

Uses Anthropic's Messages API with MCP connector and optional tool search.
Supports tool_discovery="regex" or "bm25" for advanced tool search.
"""

import os
from typing import TYPE_CHECKING, Any

import anthropic

from inspect_ai.solver._task_state import TaskState
from inspect_ai.model._providers.anthropic import AnthropicAPI

from openbench.evals.remote_mcp.base import RemoteMCPHandler

if TYPE_CHECKING:
    from inspect_ai.model._model import ModelAPI

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_PROGRESSIVE_MCP_BASE = (
    "https://progressive-mcp-bench.groq-dev.workers.dev/mcp"
)

TOOL_SEARCH_TYPES = {
    "regex": {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex",
    },
    "bm25": {
        "type": "tool_search_tool_bm25_20251119",
        "name": "tool_search_tool_bm25",
    },
}


class AnthropicRemoteMCPHandler(RemoteMCPHandler):
    """Handler for Anthropic's MCP connector with optional tool search."""

    @classmethod
    def supports_api(cls, api: "ModelAPI") -> bool:
        return isinstance(api, AnthropicAPI)

    @classmethod
    def provider_name(cls) -> str:
        return "anthropic"

    @classmethod
    def valid_tool_discovery_options(cls) -> list[str]:
        return ["regex", "bm25"]

    def _get_beta_headers(self) -> list[str]:
        """Get required beta headers for this configuration."""
        headers = ["mcp-client-2025-11-20"]
        if self.tool_discovery in ("regex", "bm25"):
            headers.append("advanced-tool-use-2025-11-20")
        return headers

    async def execute(
        self,
        state: TaskState,
        required_servers: list[str],
        servers_config: dict,
        system_message: str,
    ) -> TaskState:
        model_id = self.model_name

        api_key = os.environ.get(ANTHROPIC_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Anthropic API key not found in environment variable "
                f"{ANTHROPIC_API_KEY_ENV}. Required for 'minimal-servers-remote' "
                f"strategy with Anthropic."
            )

        mcp_servers: list[dict[str, Any]] = []
        tools: list[dict[str, Any]] = []

        for server_name in required_servers:
            server_spec: dict[str, Any] = {
                "type": "url",
                "url": f"{ANTHROPIC_PROGRESSIVE_MCP_BASE}/{server_name}",
                "name": server_name,
            }
            mcp_servers.append(server_spec)

            toolset_spec: dict[str, Any] = {
                "type": "mcp_toolset",
                "mcp_server_name": server_name,
            }
            if self.tool_discovery:
                toolset_spec["default_config"] = {"defer_loading": True}

            tools.append(toolset_spec)

        if self.tool_discovery:
            tool_search_spec = TOOL_SEARCH_TYPES.get(self.tool_discovery)
            if tool_search_spec:
                tools.append(tool_search_spec)

        user_text = state.input_text

        messages = [{"role": "user", "content": user_text}]

        beta_headers = self._get_beta_headers()

        request_payload = {
            "model": model_id,
            "max_tokens": 2048,
            "system": system_message,
            "messages": messages,
            "mcp_servers": mcp_servers,
            "tools": tools,
            "betas": beta_headers,
        }

        client = anthropic.AsyncAnthropic(api_key=api_key)

        try:
            response = await client.beta.messages.create(
                model=model_id,
                max_tokens=2048,
                system=system_message,
                messages=messages,  # type: ignore[arg-type]
                mcp_servers=mcp_servers,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                betas=beta_headers,
            )
        except Exception as e:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "api_error"
            state.metadata["error_message"] = str(e)
            state.metadata["anthropic_mcp_request"] = request_payload
            if state.output:
                state.output.completion = f"API error: {str(e)}"
            return state

        response_dict = response.model_dump() if hasattr(response, "model_dump") else {}

        assistant_text_chunks: list[str] = []
        tool_use_blocks: list[dict[str, Any]] = []
        mcp_events: list[dict[str, Any]] = []

        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                assistant_text_chunks.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                tool_use_blocks.append(
                    block.model_dump() if hasattr(block, "model_dump") else {}
                )
            elif block_type in ("server_tool_use", "tool_search_tool_result"):
                mcp_events.append(
                    block.model_dump() if hasattr(block, "model_dump") else {}
                )

        assistant_text = "".join(assistant_text_chunks).strip()

        state.metadata = state.metadata or {}
        state.metadata["remote_mcp_provider"] = "anthropic"
        state.metadata["anthropic_tool_search"] = self.tool_discovery
        state.metadata["anthropic_mcp_request"] = request_payload
        state.metadata["anthropic_mcp_response"] = response_dict
        state.metadata["anthropic_mcp_events"] = mcp_events
        state.metadata["anthropic_tool_use_blocks"] = tool_use_blocks

        if response.stop_reason == "tool_use" and tool_use_blocks:
            state.metadata["execution_note"] = (
                "Response ended with pending tool_use. For single-shot evaluation, "
                "MCP tools should be executed server-side. Check if mcp_servers "
                "are being called correctly."
            )

        if state.output:
            state.output.completion = assistant_text
        else:
            state.output = type("Output", (), {"completion": assistant_text})()

        return state
