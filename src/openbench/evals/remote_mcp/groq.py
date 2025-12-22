"""
Groq remote MCP handler.

Uses Groq's Responses API with server-side MCP support.
Supports tool_discovery="directory" for deferred tool loading.
"""

import os
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from inspect_ai.solver._task_state import TaskState

from openbench.evals.remote_mcp.base import RemoteMCPHandler
from openbench.model._providers.groq import GroqAPI

if TYPE_CHECKING:
    from inspect_ai.model._model import ModelAPI

GROQ_RESPONSES_BASE_URL = (
    os.environ.get("GROQ_BASE_URL", "https://api.groq.com").rstrip("/") + "/openai/v1"
)
GROQ_PROGRESSIVE_MCP_BASE = "https://progressive-mcp-bench.groq-dev.workers.dev/mcp"
GROQ_API_KEY_ENV = "GROQ_API_KEY"


class GroqRemoteMCPHandler(RemoteMCPHandler):
    """Handler for Groq's server-side MCP via Responses API."""

    @classmethod
    def supports_api(cls, api: "ModelAPI") -> bool:
        return isinstance(api, GroqAPI)

    @classmethod
    def provider_name(cls) -> str:
        return "groq"

    @classmethod
    def valid_tool_discovery_options(cls) -> list[str]:
        return ["directory"]

    async def execute(
        self,
        state: TaskState,
        required_servers: list[str],
        servers_config: dict,
        system_message: str,
    ) -> TaskState:
        groq_model_id = self.model_name

        mcp_tools: list[dict[str, Any]] = []
        for server_name in required_servers:
            server_desc = ""
            if server_name in servers_config:
                server_desc = servers_config[server_name].get("description", "")
            if not server_desc:
                server_desc = f"MCP server '{server_name}' for ProgressiveMCPBench"

            tool_spec: dict[str, Any] = {
                "type": "mcp",
                "server_label": server_name,
                "server_url": f"{GROQ_PROGRESSIVE_MCP_BASE}/{server_name}",
                "require_approval": "never",
                "server_description": server_desc,
            }

            if self.tool_discovery == "directory":
                tool_spec["deferred_mode"] = "directory"

            mcp_tools.append(tool_spec)

        user_text = state.input_text

        groq_input = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_text},
        ]

        api_key = os.environ.get(GROQ_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"GROQ API key not found in environment variable {GROQ_API_KEY_ENV}. "
                f"Required for 'minimal-servers-remote' strategy with Groq."
            )

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=GROQ_RESPONSES_BASE_URL,
        )

        request_payload = {
            "model": groq_model_id,
            "input": groq_input,
            "tools": mcp_tools,
            "stream": False,
            "max_output_tokens": 2048,
            "temperature": 0.7,
        }

        try:
            response = await client.responses.create(
                model=groq_model_id,
                input=groq_input,  # type: ignore[arg-type]
                tools=mcp_tools,  # type: ignore[arg-type]
                stream=False,
                max_output_tokens=2048,
                temperature=0.7,
            )
        except Exception as e:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "api_error"
            state.metadata["error_message"] = str(e)
            state.metadata["groq_mcp_request"] = request_payload
            if state.output:
                state.output.completion = f"API error: {str(e)}"
            return state
        finally:
            await client.close()

        response_dict = response.model_dump() if hasattr(response, "model_dump") else {}

        output_items = getattr(response, "output", []) or []

        assistant_text_chunks: list[str] = []
        tool_call_detected = False
        mcp_events: list[dict[str, Any]] = []

        for item in output_items:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        assistant_text_chunks.append(getattr(c, "text", ""))
            elif item_type == "function_call":
                tool_call_detected = True
            elif item_type.startswith("mcp_"):
                mcp_events.append(
                    item.model_dump()
                    if hasattr(item, "model_dump")
                    else {"type": item_type}
                )

        assistant_text = "".join(assistant_text_chunks).strip()

        state.metadata = state.metadata or {}
        state.metadata["remote_mcp_provider"] = "groq"
        state.metadata["groq_deferred_mode"] = self.tool_discovery
        state.metadata["groq_mcp_request"] = request_payload
        state.metadata["groq_mcp_response"] = response_dict
        state.metadata["groq_mcp_events"] = mcp_events

        if tool_call_detected:
            state.metadata["execution_error"] = "tool_calls_not_allowed"
            state.metadata["error_message"] = (
                "Groq Responses produced tool calls for 'minimal-servers-remote'. "
                "This strategy requires single-shot completion - the remote MCP "
                "server should handle all tool execution internally."
            )

        if state.output:
            state.output.completion = assistant_text
        else:
            state.output = type("Output", (), {"completion": assistant_text})()

        return state
