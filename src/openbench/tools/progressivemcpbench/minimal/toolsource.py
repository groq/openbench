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


def _load_tools_data() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Load config and tools data from cache.

    Returns:
        Tuple of (config, tools_by_server) where tools_by_server maps
        server_name -> list of tool info dicts with name, description, inputSchema
    """
    config, _ = get_clean_config_cached()
    raw_tools_data, _ = get_tools_json_cached()

    # Parse the nested tools.json structure into a flat mapping
    # tools.json format: list of entries with: name, description, tools: {server_name: {tools: [...]}}
    tools_by_server: dict[str, list[dict[str, Any]]] = {}
    for entry in raw_tools_data:
        tools_dict = entry.get("tools", {})
        for server_name, server_info in tools_dict.items():
            if not server_name:
                continue
            if server_name not in tools_by_server:
                tools_by_server[server_name] = []
            for tool in server_info.get("tools", []):
                tool_name = tool.get("name", "")
                if tool_name:
                    tools_by_server[server_name].append(
                        {
                            "name": tool_name,
                            "description": tool.get("description", ""),
                            "inputSchema": tool.get("inputSchema", {}),
                        }
                    )

    return config, tools_by_server


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
    config, tools_by_server = _load_tools_data()
    mcp_servers = config.get("mcpServers", {})

    # Build server index
    servers: dict[str, Server] = {}
    for name, config_data in mcp_servers.items():
        if name in required_servers:
            servers[name] = _build_server(name, config_data)

    # Build tools list from matching servers
    tools: list[Tool] = []
    for server_name in required_servers:
        if server_name not in servers:
            continue
        if server_name not in tools_by_server:
            continue

        server = servers[server_name]
        for tool_info in tools_by_server[server_name]:
            tool = create_mcp_tool_wrapper(
                server=server,
                tool_name=tool_info["name"],
                tool_description=tool_info.get("description", ""),
                input_schema=tool_info.get("inputSchema", {}),
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
    config, tools_by_server = _load_tools_data()
    mcp_servers = config.get("mcpServers", {})

    # Build set for quick lookup
    required_set = {(s, t) for s, t in required_tools}
    required_servers_set = {s for s, t in required_tools}

    # Build server index
    servers: dict[str, Server] = {}
    for name, config_data in mcp_servers.items():
        if name in required_servers_set:
            servers[name] = _build_server(name, config_data)

    # Build tools list from matching tools
    tools: list[Tool] = []
    for server_name, tool_list in tools_by_server.items():
        if server_name not in servers:
            continue

        server = servers[server_name]
        for tool_info in tool_list:
            tool_name = tool_info["name"]
            if (server_name, tool_name) not in required_set:
                continue

            tool = create_mcp_tool_wrapper(
                server=server,
                tool_name=tool_name,
                tool_description=tool_info.get("description", ""),
                input_schema=tool_info.get("inputSchema", {}),
            )
            tools.append(tool)

    return _StaticToolSource(tools)


def _distraction_tool_source(
    required_tools: list[tuple[str, str]],
    task_id: str,
    target_count: int,
    base_distractor_count: int | None = None,
) -> ToolSource:
    """Create a ToolSource with required tools plus distractors to reach target count.

    This starts with the minimal required tools and adds random irrelevant tools
    to reach the target count. This ensures a consistent baseline regardless of
    how many tools any individual server might have.

    The distraction tools are selected deterministically based on the task_id,
    ensuring reproducibility across runs.

    When base_distractor_count is provided, the first base_distractor_count distractors
    are selected using the task_id-based hash, and additional distractors are selected
    using a different hash to ensure the base set is a strict subset. This allows
    distraction-128 to be a strict superset of distraction-64.

    Args:
        required_tools: List of (server_name, tool_name) tuples for required tools
        task_id: Task identifier used for deterministic distractor selection
        target_count: Total number of tools to include
        base_distractor_count: If provided, the first N distractors use the base hash,
            remaining distractors use an extended hash for superset consistency

    Returns:
        ToolSource with required tools plus deterministic distractors
    """
    config, tools_by_server = _load_tools_data()
    mcp_servers = config.get("mcpServers", {})

    # Build all servers
    all_servers: dict[str, Server] = {}
    for name, config_data in mcp_servers.items():
        all_servers[name] = _build_server(name, config_data)

    # Build set for quick lookup of required tools
    required_set = {(s, t) for s, t in required_tools}

    # First, collect required tools (minimal tools)
    required_tools_list: list[tuple[str, dict[str, Any]]] = []
    for server_name, tool_list in tools_by_server.items():
        for tool_info in tool_list:
            tool_name = tool_info.get("name", "")
            if (server_name, tool_name) in required_set:
                required_tools_list.append((server_name, tool_info))

    # Calculate how many distractors we need
    total_distractors_needed = target_count - len(required_tools_list)

    # Collect all non-required tools as potential distractors
    distractor_pool: list[tuple[str, dict[str, Any]]] = []
    for server_name, tool_list in tools_by_server.items():
        for tool_info in tool_list:
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
    if base_distractor_count is not None and total_distractors_needed > base_distractor_count:
        # Take the first base_distractor_count using the base hash (same as distraction-64)
        base_distractors = distractor_pool[:base_distractor_count]

        # For additional distractors, use a different hash to select from remaining pool
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
        # Simple case: just take from the sorted pool
        selected_distractors = distractor_pool[:total_distractors_needed] if total_distractors_needed > 0 else []

    # Combine required and distractor tools
    all_selected = required_tools_list + selected_distractors

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
            input_schema=tool_info.get("inputSchema", {}),
        )
        tools.append(tool)

    return _StaticToolSource(tools)


# Base distractor count for distraction-64 (used for superset consistency)
_DISTRACTION_64_BASE = 64


def distraction_64_tool_source(
    required_tools: list[tuple[str, str]],
    task_id: str,
) -> ToolSource:
    """Create a ToolSource with required tools plus distractors to reach 64 tools.

    Args:
        required_tools: List of (server_name, tool_name) tuples for required tools
        task_id: Task identifier used for deterministic distractor selection

    Returns:
        ToolSource with required tools plus deterministic distractors (64 total)
    """
    return _distraction_tool_source(required_tools, task_id, target_count=64)


def distraction_128_tool_source(
    required_tools: list[tuple[str, str]],
    task_id: str,
) -> ToolSource:
    """Create a ToolSource with required tools plus distractors to reach 128 tools.

    This is a strict superset of distraction-64: it contains all 64 tools from
    distraction-64 plus an additional 64 deterministically selected tools.

    Args:
        required_tools: List of (server_name, tool_name) tuples for required tools
        task_id: Task identifier used for deterministic distractor selection

    Returns:
        ToolSource with required tools plus deterministic distractors (128 total)
    """
    # Calculate how many distractors distraction-64 would use for this task
    # We need to match its base selection, then add more
    base_distractors = _DISTRACTION_64_BASE - len(required_tools)
    return _distraction_tool_source(
        required_tools, task_id, target_count=128, base_distractor_count=max(0, base_distractors)
    )


class _StaticToolSource(ToolSource):
    """Simple ToolSource that returns a static list of tools."""

    def __init__(self, tools: list[Tool]):
        self._tools = tools

    async def tools(self) -> list[Tool]:
        return self._tools
