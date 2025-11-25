"""
ToolSource factories for minimal ProgressMCPBench strategies.

These strategies bypass the discovery phase and provide tools directly,
based on task annotations in the dataset.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from inspect_ai.tool import Tool, ToolSource

from ..copilot.upstream_cache import get_tools_json_cached, get_clean_config_cached
from ..copilot.schemas import Server, ServerConfig
from .tool_wrapper import create_mcp_tool_wrapper


def _root_sandbox_dir() -> Path:
    return Path(os.path.expanduser("~/.openbench/progressivemcpbench/root")).resolve()


def _load_tools_data() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load config and tools data from cache."""
    config, _ = get_clean_config_cached()
    tools_data, _ = get_tools_json_cached()
    return config, tools_data


def _build_server(name: str, config_data: dict[str, Any]) -> Server:
    """Build a Server object with path rewrites applied."""
    sandbox = _root_sandbox_dir()
    cfg = dict(config_data)
    args = list(cfg.get("args", []))

    if name.lower() in {
        "filesystem",
        "server-filesystem",
        "@modelcontextprotocol/server-filesystem",
    }:
        if args and args[-1] == "/":
            args[-1] = str(sandbox)

    for i, a in enumerate(args):
        if isinstance(a, str) and "annotated_data" in a:
            parts = a.split("annotated_data/", 1)
            if len(parts) == 2:
                args[i] = str(sandbox / parts[1])

    cfg["args"] = args
    return Server(name=name, config=ServerConfig(**cfg))


def minimal_servers_tool_source(
    required_servers: list[str],
) -> ToolSource:
    """Create a ToolSource with all tools from the specified servers.

    Args:
        required_servers: List of server names to include tools from

    Returns:
        ToolSource with all tools from the specified servers
    """
    config, tools_data = _load_tools_data()
    mcp_servers = config.get("mcpServers", {})

    # Build server index and tools index
    servers: dict[str, Server] = {}
    for name, config_data in mcp_servers.items():
        if name in required_servers:
            servers[name] = _build_server(name, config_data)

    # Build tools list from matching servers
    tools: list[Tool] = []
    for server_data in tools_data:
        server_name = server_data.get("server_name", "")
        if server_name not in required_servers:
            continue
        if server_name not in servers:
            continue

        server = servers[server_name]
        for tool_info in server_data.get("tools", []):
            tool_name = tool_info.get("name", "")
            if not tool_name:
                continue

            tool = create_mcp_tool_wrapper(
                server=server,
                tool_name=tool_name,
                tool_description=tool_info.get("description", ""),
                input_schema=tool_info.get("parameter", {}),
            )
            tools.append(tool)

    return _StaticToolSource(tools)


def minimal_tools_tool_source(
    required_tools: list[tuple[str, str]],
) -> ToolSource:
    """Create a ToolSource with only the specified tools.

    Args:
        required_tools: List of (server_name, tool_name) tuples

    Returns:
        ToolSource with only the specified tools
    """
    config, tools_data = _load_tools_data()
    mcp_servers = config.get("mcpServers", {})

    # Build set for quick lookup
    required_set = {(s, t) for s, t in required_tools}
    required_servers = {s for s, t in required_tools}

    # Build server index
    servers: dict[str, Server] = {}
    for name, config_data in mcp_servers.items():
        if name in required_servers:
            servers[name] = _build_server(name, config_data)

    # Build tools list from matching tools
    tools: list[Tool] = []
    for server_data in tools_data:
        server_name = server_data.get("server_name", "")
        if server_name not in servers:
            continue

        server = servers[server_name]
        for tool_info in server_data.get("tools", []):
            tool_name = tool_info.get("name", "")
            if (server_name, tool_name) not in required_set:
                continue

            tool = create_mcp_tool_wrapper(
                server=server,
                tool_name=tool_name,
                tool_description=tool_info.get("description", ""),
                input_schema=tool_info.get("parameter", {}),
            )
            tools.append(tool)

    return _StaticToolSource(tools)


def distraction_128_tool_source(
    required_servers: list[str],
    task_id: str,
    target_count: int = 128,
) -> ToolSource:
    """Create a ToolSource with required tools plus distractors to reach target count.

    The distraction tools are selected deterministically based on the task_id,
    ensuring reproducibility across runs.

    Args:
        required_servers: List of server names that must be included
        task_id: Task identifier used for deterministic distractor selection
        target_count: Total number of tools to include (default 128)

    Returns:
        ToolSource with required tools plus deterministic distractors
    """
    config, tools_data = _load_tools_data()
    mcp_servers = config.get("mcpServers", {})

    # Build all servers
    all_servers: dict[str, Server] = {}
    for name, config_data in mcp_servers.items():
        all_servers[name] = _build_server(name, config_data)

    # Collect all tools grouped by server
    all_tools_by_server: dict[str, list[dict[str, Any]]] = {}
    for server_data in tools_data:
        server_name = server_data.get("server_name", "")
        if server_name in all_servers:
            all_tools_by_server[server_name] = server_data.get("tools", [])

    # First, collect required tools
    required_tools: list[tuple[str, dict[str, Any]]] = []
    for server_name in required_servers:
        if server_name in all_tools_by_server:
            for tool_info in all_tools_by_server[server_name]:
                required_tools.append((server_name, tool_info))

    # Calculate how many distractors we need
    remaining = target_count - len(required_tools)

    # Collect all non-required tools as potential distractors
    distractor_pool: list[tuple[str, dict[str, Any]]] = []
    for server_name, tool_list in all_tools_by_server.items():
        if server_name not in required_servers:
            for tool_info in tool_list:
                distractor_pool.append((server_name, tool_info))

    # Sort distractors deterministically based on task_id hash
    def sort_key(item: tuple[str, dict[str, Any]]) -> str:
        server_name, tool_info = item
        tool_name = tool_info.get("name", "")
        combined = f"{task_id}:{server_name}:{tool_name}"
        return hashlib.sha256(combined.encode()).hexdigest()

    distractor_pool.sort(key=sort_key)

    # Select distractors
    selected_distractors = distractor_pool[:remaining] if remaining > 0 else []

    # Combine required and distractor tools
    all_selected = required_tools + selected_distractors

    # Build Tool objects
    tools: list[Tool] = []
    for server_name, tool_info in all_selected:
        if server_name not in all_servers:
            continue

        server = all_servers[server_name]
        tool_name = tool_info.get("name", "")
        if not tool_name:
            continue

        tool = create_mcp_tool_wrapper(
            server=server,
            tool_name=tool_name,
            tool_description=tool_info.get("description", ""),
            input_schema=tool_info.get("parameter", {}),
        )
        tools.append(tool)

    return _StaticToolSource(tools)


class _StaticToolSource(ToolSource):
    """Simple ToolSource that returns a static list of tools."""

    def __init__(self, tools: list[Tool]):
        self._tools = tools

    async def tools(self) -> list[Tool]:
        return self._tools
