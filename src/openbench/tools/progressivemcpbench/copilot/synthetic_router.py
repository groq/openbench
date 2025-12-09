"""
Synthetic Router for Copilot strategy using HTTP MCP backend.

This router uses the same semantic matching (ToolMatcher) as the live copilot,
but executes tools via the synthetic HTTP MCP server instead of real MCP servers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import mcp.types as types

from .matcher import ToolMatcher

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)


DEFAULT_HTTP_HOST = "localhost"
DEFAULT_HTTP_PORT = 9123


def dump_to_yaml(data: dict[str, Any]) -> str:
    """Serialize a dict to YAML (fallback to JSON if PyYAML is unavailable)."""
    if yaml is not None:
        return yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return json.dumps(data, indent=2, ensure_ascii=False)


def _synthetic_mcp_dir() -> Path:
    """Get the synthetic_mcp directory path."""
    current = Path(__file__).resolve()
    repo_root = current.parent.parent.parent.parent.parent.parent
    return repo_root / "synthetic_mcp"


def _call_http_tool(
    server_name: str,
    tool_name: str,
    params: dict[str, Any],
    host: str = DEFAULT_HTTP_HOST,
    port: int = DEFAULT_HTTP_PORT,
    timeout: int = 30,
) -> dict[str, Any]:
    """Call a tool on the synthetic HTTP MCP server using MCP protocol."""
    conn = HTTPConnection(host, port, timeout=timeout)

    # MCP protocol format
    call_data = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": params
        }
    }
    body = json.dumps(call_data)

    try:
        conn.request(
            "POST",
            f"/mcp/{server_name}",
            body,
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
        )
        response = conn.getresponse()
        response_text = response.read().decode("utf-8")

        # Parse SSE response format: "event: message\ndata: {...}\n\n"
        try:
            # Extract JSON from data field in SSE
            if "data: " in response_text:
                # Split on event/data boundaries
                lines = response_text.strip().split('\n')
                data_line = None
                for line in lines:
                    if line.startswith('data: '):
                        data_line = line[6:]  # Remove "data: " prefix
                        break

                if data_line:
                    data = json.loads(data_line)
                else:
                    return {"error": "No data field in SSE response"}
            else:
                # Try to parse as direct JSON
                data = json.loads(response_text)

            if "error" in data:
                return {"error": data["error"].get("message", str(data["error"]))}

            return data

        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse response: {str(e)}"}

    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


class SyntheticCopilotRouter:
    """Router that uses ToolMatcher for semantic search and HTTP backend for execution."""

    def __init__(
        self,
        mcp_arg_path: Path,
        http_host: str = DEFAULT_HTTP_HOST,
        http_port: int = DEFAULT_HTTP_PORT,
        timeout: int = 30,
    ):
        """Initialize the synthetic copilot router.

        Args:
            mcp_arg_path: Path to the mcp_arg_*.json embeddings file
            http_host: HTTP MCP server host
            http_port: HTTP MCP server port
            timeout: Timeout for HTTP requests
        """
        self.http_host = http_host
        self.http_port = http_port
        self.timeout = timeout

        # Initialize ToolMatcher with same settings as live router
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        embedding_dims = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        top_servers = int(os.getenv("TOP_SERVERS", "5"))
        top_tools = int(os.getenv("TOP_TOOLS", "3"))

        self.matcher = ToolMatcher(
            embedding_model=embedding_model,
            dimensions=embedding_dims,
            top_servers=top_servers,
            top_tools=top_tools,
        )

        # Setup OpenAI client for embeddings
        base_url = os.getenv("EMBEDDING_BASE_URL")
        api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY or EMBEDDING_API_KEY for routing embeddings."
            )
        if not mcp_arg_path.exists():
            raise ValueError(f"MCP embeddings file not found at: {mcp_arg_path}")

        self.matcher.setup_openai_client(base_url=base_url, api_key=api_key)  # type: ignore[arg-type]
        self.matcher.load_data(str(mcp_arg_path))

        # Lock for connection management
        self.connection_lock = asyncio.Lock()

    async def route(self, query: str) -> dict[str, Any]:
        """Route using ToolMatcher to find the best tools."""
        return self.matcher.match(query)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        params: dict[str, Any] | None = None,
        timeout: int = 300,
    ) -> types.CallToolResult:
        """Execute a tool via the synthetic HTTP MCP server."""

        async with self.connection_lock:
            try:
                response = _call_http_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    params=params or {},
                    host=self.http_host,
                    port=self.http_port,
                    timeout=min(timeout, self.timeout),
                )

                if "error" in response and response["error"]:
                    return types.CallToolResult(
                        isError=True,
                        content=[
                            types.TextContent(
                                type="text", text=f"Error: {response['error']}"
                            )
                        ],
                    )

                result = response.get("result", "(no result)")
                if isinstance(result, str):
                    text = result
                else:
                    text = json.dumps(result, indent=2)

                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=text)]
                )

            except asyncio.TimeoutError:
                return types.CallToolResult(
                    isError=True,
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"Tool {tool_name} in {server_name} call timed out.",
                        )
                    ],
                )
            except Exception as e:
                error_msg = (
                    f"Error executing tool {tool_name} on {server_name}: {str(e)}"
                )
                return types.CallToolResult(
                    isError=True,
                    content=[
                        types.TextContent(
                            type="text",
                            text=error_msg,
                        )
                    ],
                )

    async def aclose(self) -> None:
        pass

    async def __aenter__(self) -> "SyntheticCopilotRouter":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()
