"""
ToolSource factories for synthetic MCP strategies.

These strategies load tool definitions from the synthetic servers.json config
and route tool calls to the HTTP MCP server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai.tool import Tool, ToolSource

from .tool_wrapper import (
    create_synthetic_tool_wrapper,
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PORT,
)


def _synthetic_mcp_dir() -> Path:
    """Get the synthetic_mcp directory path."""
    # Navigate from src/openbench/tools/progressivemcpbench/synthetic/
    # to repo root, then to synthetic_mcp/
    current = Path(__file__).resolve()
    # Go up: synthetic -> minimal -> progressivemcpbench -> tools -> openbench -> src -> repo_root
    repo_root = current.parent.parent.parent.parent.parent.parent
    return repo_root / "synthetic_mcp"


def _load_servers_config() -> dict[str, Any]:
    """Load the synthetic servers.json configuration."""
    config_path = _synthetic_mcp_dir() / "config" / "servers.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _load_synthetic_tasks() -> list[dict[str, Any]]:
    """Load the synthetic tasks JSON."""
    tasks_path = _synthetic_mcp_dir() / "tasks" / "progressivemcpbench_synthetic.json"
    with open(tasks_path, encoding="utf-8") as f:
        return json.load(f)


class _StaticToolSource(ToolSource):
    """Simple ToolSource that returns a static list of tools."""

    def __init__(self, tools: list[Tool]):
        self._tools = tools

    async def tools(self) -> list[Tool]:
        return self._tools


def synthetic_minimal_servers_tool_source(
    required_servers: list[str],
    http_host: str = DEFAULT_HTTP_HOST,
    http_port: int = DEFAULT_HTTP_PORT,
) -> ToolSource:
    """Create a ToolSource with all tools from the specified synthetic servers.

    Args:
        required_servers: List of server names to include tools from
        http_host: HTTP MCP server host
        http_port: HTTP MCP server port

    Returns:
        ToolSource with all tools from the specified servers
    """
    servers_config = _load_servers_config()

    tools: list[Tool] = []
    for server_name in required_servers:
        if server_name not in servers_config:
            continue

        server = servers_config[server_name]
        for tool_info in server.get("tools", []):
            tool = create_synthetic_tool_wrapper(
                server_name=server_name,
                tool_name=tool_info["name"],
                tool_description=tool_info.get("description", ""),
                input_schema=tool_info.get("inputSchema", {}),
                http_host=http_host,
                http_port=http_port,
            )
            tools.append(tool)

    return _StaticToolSource(tools)


def synthetic_minimal_tools_tool_source(
    required_tools: list[tuple[str, str]],
    http_host: str = DEFAULT_HTTP_HOST,
    http_port: int = DEFAULT_HTTP_PORT,
) -> ToolSource:
    """Create a ToolSource with only the specified synthetic tools.

    Args:
        required_tools: List of (server_name, tool_name) tuples
        http_host: HTTP MCP server host
        http_port: HTTP MCP server port

    Returns:
        ToolSource with only the specified tools
    """
    servers_config = _load_servers_config()

    required_set = {(s, t) for s, t in required_tools}
    required_servers_set = {s for s, t in required_tools}

    tools: list[Tool] = []
    for server_name in required_servers_set:
        if server_name not in servers_config:
            continue

        server = servers_config[server_name]
        for tool_info in server.get("tools", []):
            tool_name = tool_info["name"]
            if (server_name, tool_name) not in required_set:
                continue

            tool = create_synthetic_tool_wrapper(
                server_name=server_name,
                tool_name=tool_name,
                tool_description=tool_info.get("description", ""),
                input_schema=tool_info.get("inputSchema", {}),
                http_host=http_host,
                http_port=http_port,
            )
            tools.append(tool)

    return _StaticToolSource(tools)


def synthetic_all_tools_tool_source(
    http_host: str = DEFAULT_HTTP_HOST,
    http_port: int = DEFAULT_HTTP_PORT,
) -> ToolSource:
    """Create a ToolSource with all tools from all synthetic servers.

    This is useful for testing or for strategies that want access to
    all available tools.

    Args:
        http_host: HTTP MCP server host
        http_port: HTTP MCP server port

    Returns:
        ToolSource with all available synthetic tools
    """
    servers_config = _load_servers_config()

    tools: list[Tool] = []
    for server_name, server in servers_config.items():
        for tool_info in server.get("tools", []):
            tool = create_synthetic_tool_wrapper(
                server_name=server_name,
                tool_name=tool_info["name"],
                tool_description=tool_info.get("description", ""),
                input_schema=tool_info.get("inputSchema", {}),
                http_host=http_host,
                http_port=http_port,
            )
            tools.append(tool)

    return _StaticToolSource(tools)
