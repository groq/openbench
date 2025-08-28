"""
Generic MCP Tools Integration for OpenBench

This module provides a generic, reusable framework for integrating MCP (Model Context Protocol)
tools into OpenBench evaluations using inspect.ai's native MCP support.

This approach can be used for any evaluation that needs MCP tools - just provide:
1. A tools.json file with tool definitions
2. An all_config.json file with server configurations

Usage:
    from openbench.tools.livemcpbench.mcp_tools import get_mcp_tool_sources

    # Basic usage
    tool_sources = get_mcp_tool_sources(
        tools_json_path="path/to/tools.json",
        config_json_path="path/to/all_config.json"
    )

    # Use in evaluation
    solver = react(tools=tool_sources)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Literal
from inspect_ai.tool import mcp_tools, mcp_server_stdio, ToolSource

from .data_fetcher import get_tools_data, get_config_data

logger = logging.getLogger(__name__)


class MCPToolsRegistry:
    """
    Generic registry for MCP tools using inspect.ai's native MCP support.

    This class loads MCP server configurations and tool definitions from
    JSON files and creates ToolSource objects that can be directly used
    with inspect.ai's react solver.

    Can be used for any MCP-based evaluation by providing the appropriate
    tools.json and config.json files.
    """

    def __init__(
        self,
        tools_json_path: Optional[Path] = None,
        config_json_path: Optional[Path] = None,
    ):
        """
        Initialize the MCP tools registry.

        Args:
            tools_json_path: Path to tools.json file with tool definitions (optional, uses dynamic fetching if None)
            config_json_path: Path to config.json file with server configurations (optional, uses dynamic fetching if None)
        """
        self.tools_json_path = tools_json_path
        self.config_json_path = config_json_path

        # Storage for configurations
        self._tools_data: List[Dict[str, Any]] = []
        self._server_configs: List[Dict[str, Any]] = []
        self._tools_by_category: Dict[str, List[Dict[str, Any]]] = {}

        self._problem_servers: List[str] = [
            "text-editor",  # has known schema validation issues
            "searxng",  # File Access category - prone to validation issues
            "okppt",  # requires pycairo which needs Cairo graphics library
            "chess",  # requires cairosvg/cairocffi which can't find Cairo in uvx environment
            "filesystem",  # has directory path configuration issues (empty path)
            "drawing",  # server crashes/disconnects causing BrokenResourceError
        ]

        # Load configurations
        self._load_configurations()
        self._categorize_tools()

    def _load_configurations(self) -> None:
        """Load tool and server configurations from JSON files or dynamic fetching."""
        try:
            # Load tools configuration
            if self.tools_json_path is not None:
                # Use provided file path
                if self.tools_json_path.exists():
                    with open(self.tools_json_path, "r", encoding="utf-8") as f:
                        self._tools_data = json.load(f)
                    logger.info(
                        f"Loaded {len(self._tools_data)} tool definitions from {self.tools_json_path}"
                    )
                else:
                    logger.error(f"Tools JSON file not found: {self.tools_json_path}")
                    raise FileNotFoundError(
                        f"Tools JSON file not found: {self.tools_json_path}"
                    )
            else:
                # Use dynamic fetching
                self._tools_data = get_tools_data()
                logger.info(
                    f"Loaded {len(self._tools_data)} tool definitions from dynamic fetch"
                )

            # Load server configuration
            if self.config_json_path is not None:
                # Use provided file path
                if self.config_json_path.exists():
                    with open(self.config_json_path, "r", encoding="utf-8") as f:
                        self._server_configs = json.load(f)
                    logger.info(
                        f"Loaded {len(self._server_configs)} server configurations from {self.config_json_path}"
                    )
                else:
                    logger.error(f"Config JSON file not found: {self.config_json_path}")
                    raise FileNotFoundError(
                        f"Config JSON file not found: {self.config_json_path}"
                    )
            else:
                # Use dynamic fetching
                self._server_configs = get_config_data()
                logger.info(
                    f"Loaded {len(self._server_configs)} server configurations from dynamic fetch"
                )

        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise

    def _categorize_tools(self) -> None:
        """Categorize tools by their function/category."""
        self._tools_by_category.clear()

        for tool_data in self._tools_data:
            category = tool_data.get("category", "Miscellaneous")
            if category not in self._tools_by_category:
                self._tools_by_category[category] = []
            self._tools_by_category[category].append(tool_data)

        logger.info(
            f"Categorized tools into {len(self._tools_by_category)} categories: {list(self._tools_by_category.keys())}"
        )

    def get_categories(self) -> List[str]:
        """Get all available tool categories."""
        return list(self._tools_by_category.keys())

    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all tools in a specific category."""
        return self._tools_by_category.get(category, [])

    def create_tool_sources(
        self,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
        tools_per_server: Union[str, List[str]] = "all",
    ) -> List[ToolSource]:
        """
        Create ToolSource objects for MCP servers using inspect.ai's native support.

        Args:
            categories: List of tool categories to include (e.g., ["Finance", "Discovery"])
            limit: Maximum number of servers per category to include
            tools_per_server: Either "all" to include all tools, or list of specific tool names

        Returns:
            List of ToolSource objects ready for use with inspect.ai
        """
        tool_sources = []
        processed_servers = set()  # Avoid duplicate servers

        # Determine which categories to process
        target_categories = categories or self.get_categories()

        for category in target_categories:
            tools_in_category = self.get_tools_by_category(category)

            # Apply limit if specified
            if limit:
                tools_in_category = tools_in_category[:limit]

            for tool_spec in tools_in_category:
                # Extract server configuration from the tool spec
                server_configs = self._extract_server_configs(tool_spec)

                for server_name, server_config in server_configs.items():
                    if server_name in processed_servers:
                        continue  # Skip already processed servers

                    if server_name in self._problem_servers:
                        logger.warning(
                            f"Skipping {server_name} server due to known schema validation issues"
                        )
                        continue

                    try:
                        # Create MCP server using inspect.ai's native support
                        server_env = (
                            server_config.get("env", {}).copy()
                            if server_config.get("env")
                            else {}
                        )

                        mcp_server = mcp_server_stdio(
                            command=server_config["command"],
                            args=server_config.get("args", []),
                            cwd=server_config.get("cwd"),
                            env=server_env,
                        )

                        # Determine which tools to include from this server
                        server_tools = self._get_tools_for_server(
                            tool_spec, server_name, tools_per_server
                        )

                        # Create tool source
                        tool_source = mcp_tools(mcp_server, tools=server_tools)
                        tool_sources.append(tool_source)
                        processed_servers.add(server_name)

                        logger.info(
                            f"Created tool source for server '{server_name}' with tools: {server_tools}"
                        )

                    except Exception as e:
                        # Check if this is a known problematic error type
                        error_str = str(e)
                        if any(
                            keyword in error_str
                            for keyword in [
                                "ValidationError",
                                "literal_error",
                                "Invalid value",
                                "union",
                                "JSON Schema",
                            ]
                        ):
                            logger.warning(
                                f"Skipping server '{server_name}' due to JSON Schema validation issue: {e}"
                            )
                            # Add to problem servers list to skip in future
                            if server_name not in self._problem_servers:
                                self._problem_servers.append(server_name)
                        elif any(
                            keyword in error_str
                            for keyword in [
                                "BrokenResourceError",
                                "ClosedResourceError",
                                "ConnectionError",
                            ]
                        ):
                            logger.warning(
                                f"Skipping server '{server_name}' due to connection/resource issue: {e}"
                            )
                            if server_name not in self._problem_servers:
                                self._problem_servers.append(server_name)
                        else:
                            logger.error(
                                f"Failed to create tool source for server '{server_name}': {e}"
                            )
                        continue

        logger.info(
            f"Created {len(tool_sources)} MCP tool sources from {len(processed_servers)} servers"
        )
        return tool_sources

    def _extract_server_configs(
        self, tool_spec: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract server configurations from a tool specification."""
        server_configs = {}

        # Get config from the tool spec
        config = tool_spec.get("config", {})
        mcp_servers = config.get("mcpServers", {})

        for server_name, server_config in mcp_servers.items():
            server_configs[server_name] = server_config

        return server_configs

    def _get_tools_for_server(
        self,
        tool_spec: Dict[str, Any],
        server_name: str,
        tools_per_server: Union[str, List[str]],
    ) -> Union[Literal["all"], List[str]]:
        """Get the list of tools to include for a specific server."""
        if tools_per_server == "all":
            return "all"

        if isinstance(tools_per_server, list):
            return tools_per_server

        # Extract available tools from the tool spec
        tools_data = tool_spec.get("tools", {})
        if server_name in tools_data:
            server_tools_data = tools_data[server_name]
            available_tools = [
                tool["name"] for tool in server_tools_data.get("tools", [])
            ]
            return available_tools

        return "all"  # Fallback

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the tool registry."""
        return {
            "total_tools": len(self._tools_data),
            "categories": len(self.get_categories()),
            "category_breakdown": {
                category: len(tools)
                for category, tools in self._tools_by_category.items()
            },
        }


def get_mcp_tool_sources(
    tools_json_path: Optional[Union[str, Path]] = None,
    config_json_path: Optional[Union[str, Path]] = None,
    categories: Optional[List[str]] = None,
    limit: Optional[int] = None,
    tools_per_server: Union[str, List[str]] = "all",
) -> List[ToolSource]:
    """
    Get MCP tool sources for use in inspect.ai evaluations using native MCP support.

    Args:
        tools_json_path: Path to tools.json file. If None, uses dynamic fetching from upstream
        config_json_path: Path to config.json file. If None, uses dynamic fetching from upstream
        categories: List of tool categories to include (e.g., ["Finance", "Discovery"])
        limit: Maximum number of servers per category to load (reduces installation logs)
        tools_per_server: Either "all" or list of specific tool names to include per server

    Returns:
        List of ToolSource objects ready for use in inspect.ai evaluations

    Example:
        # Use dynamic fetching (recommended)
        tool_sources = get_mcp_tool_sources(categories=["Finance"], limit=2)

        # Use custom tool definitions
        tool_sources = get_mcp_tool_sources(
            tools_json_path="my_eval/tools.json",
            config_json_path="my_eval/servers.json",
            categories=["Custom"]
        )

        # Use in a task
        solver = react(tools=tool_sources)
    """
    # Convert to Path objects if provided
    tools_path = Path(tools_json_path) if tools_json_path is not None else None
    config_path = Path(config_json_path) if config_json_path is not None else None

    # Create registry and get tool sources
    registry = MCPToolsRegistry(
        tools_json_path=tools_path, config_json_path=config_path
    )

    return registry.create_tool_sources(
        categories=categories, limit=limit, tools_per_server=tools_per_server
    )


def get_tool_categories(
    tools_json_path: Optional[Union[str, Path]] = None,
    config_json_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Get all available tool categories from the specified configuration files or dynamic fetching."""
    # Convert to Path objects if provided
    tools_path = Path(tools_json_path) if tools_json_path is not None else None
    config_path = Path(config_json_path) if config_json_path is not None else None

    registry = MCPToolsRegistry(
        tools_json_path=tools_path, config_json_path=config_path
    )

    return registry.get_categories()


def get_registry_stats(
    tools_json_path: Optional[Union[str, Path]] = None,
    config_json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Get statistics about the tool registry using dynamic fetching or specified files."""
    # Convert to Path objects if provided
    tools_path = Path(tools_json_path) if tools_json_path is not None else None
    config_path = Path(config_json_path) if config_json_path is not None else None

    registry = MCPToolsRegistry(
        tools_json_path=tools_path, config_json_path=config_path
    )

    return registry.get_stats()
