"""
ToolSource factories for synthetic MCP strategies.

These strategies load tool definitions from the synthetic servers.json config
and route tool calls to the HTTP MCP server.
"""

from __future__ import annotations

import hashlib
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
    # Navigate from src/openbench/tools/progressivemcpbench/synthetic/toolsource.py
    # to repo root (5 parents), then to synthetic_mcp/
    current = Path(__file__).resolve()
    # Go up: synthetic -> progressivemcpbench -> tools -> openbench -> src -> repo_root
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


def _synthetic_distraction_tool_source(
    required_tools: list[tuple[str, str]],
    task_id: str,
    target_count: int,
    base_distractor_count: int | None = None,
    http_host: str = DEFAULT_HTTP_HOST,
    http_port: int = DEFAULT_HTTP_PORT,
) -> ToolSource:
    """Create a ToolSource with required tools plus distractors to reach target count.

    This starts with the minimal required tools and adds random irrelevant tools
    to reach the target count. The distraction tools are selected deterministically
    based on the task_id, ensuring reproducibility across runs.

    When base_distractor_count is provided, the first base_distractor_count distractors
    are selected using the task_id-based hash, and additional distractors are selected
    using a different hash to ensure the base set is a strict subset. This allows
    distraction-128 to be a strict superset of distraction-64.

    Args:
        required_tools: List of (server_name, tool_name) tuples for required tools
        task_id: Task identifier used for deterministic distractor selection
        target_count: Total number of tools to include
        base_distractor_count: If provided, the first N distractors use the base hash
        http_host: HTTP MCP server host
        http_port: HTTP MCP server port

    Returns:
        ToolSource with required tools plus deterministic distractors
    """
    servers_config = _load_servers_config()

    # Build set for quick lookup of required tools
    required_set = {(s, t) for s, t in required_tools}

    # Collect required tools
    required_tools_list: list[tuple[str, dict[str, Any]]] = []
    for server_name, server in servers_config.items():
        for tool_info in server.get("tools", []):
            tool_name = tool_info.get("name", "")
            if (server_name, tool_name) in required_set:
                required_tools_list.append((server_name, tool_info))

    # Calculate how many distractors we need
    total_distractors_needed = target_count - len(required_tools_list)

    # Collect all non-required tools as potential distractors
    distractor_pool: list[tuple[str, dict[str, Any]]] = []
    for server_name, server in servers_config.items():
        for tool_info in server.get("tools", []):
            tool_name = tool_info.get("name", "")
            if (server_name, tool_name) not in required_set:
                distractor_pool.append((server_name, tool_info))

    # Sort distractors deterministically based on task_id hash
    def base_sort_key(item: tuple[str, dict[str, Any]]) -> str:
        server_name, tool_info = item
        tool_name = tool_info.get("name", "")
        combined = f"{task_id}:{server_name}:{tool_name}"
        return hashlib.sha256(combined.encode()).hexdigest()

    distractor_pool.sort(key=base_sort_key)

    # Select distractors with superset consistency
    if (
        base_distractor_count is not None
        and total_distractors_needed > base_distractor_count
    ):
        # Take the first base_distractor_count using the base hash
        base_distractors = distractor_pool[:base_distractor_count]

        # For additional distractors, use a different hash
        remaining_pool = distractor_pool[base_distractor_count:]
        additional_needed = total_distractors_needed - base_distractor_count

        def extended_sort_key(item: tuple[str, dict[str, Any]]) -> str:
            server_name, tool_info = item
            tool_name = tool_info.get("name", "")
            combined = f"{task_id}:extended:{server_name}:{tool_name}"
            return hashlib.sha256(combined.encode()).hexdigest()

        remaining_pool.sort(key=extended_sort_key)
        additional_distractors = remaining_pool[:additional_needed]

        selected_distractors = base_distractors + additional_distractors
    else:
        selected_distractors = (
            distractor_pool[:total_distractors_needed]
            if total_distractors_needed > 0
            else []
        )

    # Combine required and distractor tools
    all_selected = required_tools_list + selected_distractors

    # Build Tool objects
    tools: list[Tool] = []
    for server_name, tool_info in all_selected:
        tool_name = tool_info.get("name", "")
        if not tool_name:
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


# Base distractor count for distraction-64 (used for superset consistency)
_DISTRACTION_64_BASE = 64


def synthetic_distraction_64_tool_source(
    required_tools: list[tuple[str, str]],
    task_id: str,
    http_host: str = DEFAULT_HTTP_HOST,
    http_port: int = DEFAULT_HTTP_PORT,
) -> ToolSource:
    """Create a ToolSource with required tools plus distractors to reach 64 tools.

    Args:
        required_tools: List of (server_name, tool_name) tuples for required tools
        task_id: Task identifier used for deterministic distractor selection
        http_host: HTTP MCP server host
        http_port: HTTP MCP server port

    Returns:
        ToolSource with required tools plus deterministic distractors (64 total)
    """
    return _synthetic_distraction_tool_source(
        required_tools,
        task_id,
        target_count=64,
        http_host=http_host,
        http_port=http_port,
    )


def synthetic_distraction_128_tool_source(
    required_tools: list[tuple[str, str]],
    task_id: str,
    http_host: str = DEFAULT_HTTP_HOST,
    http_port: int = DEFAULT_HTTP_PORT,
) -> ToolSource:
    """Create a ToolSource with required tools plus distractors to reach 128 tools.

    This is a strict superset of distraction-64: it contains all 64 tools from
    distraction-64 plus an additional 64 deterministically selected tools.

    Args:
        required_tools: List of (server_name, tool_name) tuples for required tools
        task_id: Task identifier used for deterministic distractor selection
        http_host: HTTP MCP server host
        http_port: HTTP MCP server port

    Returns:
        ToolSource with required tools plus deterministic distractors (128 total)
    """
    base_distractors = _DISTRACTION_64_BASE - len(required_tools)
    return _synthetic_distraction_tool_source(
        required_tools,
        task_id,
        target_count=128,
        base_distractor_count=max(0, base_distractors),
        http_host=http_host,
        http_port=http_port,
    )
