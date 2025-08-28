"""
LiveMCPBench Tool Integration with Dynamic Data Fetching

This module provides a generic, reusable framework for integrating MCP tools
into OpenBench evaluations using inspect.ai's native MCP support.

Basic Usage:
    from openbench.tools.livemcpbench import get_mcp_tool_sources

    # Get LiveMCPBench tools
    tool_sources = get_mcp_tool_sources(categories=["Finance", "Discovery"], limit=3)

    # Use in evaluation
    solver = react(tools=tool_sources)
"""

from .mcp_tools import (
    get_mcp_tool_sources,
    get_tool_categories,
    get_registry_stats,
    MCPToolsRegistry,
)
from .data_fetcher import (
    get_tools_data,
    get_config_data,
    LiveMCPDataFetcher,
)


__all__ = [
    "get_mcp_tool_sources",
    "get_tool_categories",
    "get_registry_stats",
    "MCPToolsRegistry",
    "get_tools_data",
    "get_config_data",
    "clear_cache",
    "get_cache_info",
    "LiveMCPDataFetcher",
]
