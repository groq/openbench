"""
Generic MCP Tools Integration for OpenBench

This module provides a generic, reusable framework for integrating MCP (Model Context Protocol)
tools into OpenBench evaluations using inspect.ai's native MCP support.



Usage:
    from openbench.tools.livemcpbench.mcp_tools import get_mcp_tool_sources

    # Basic usage - fetches all tools from upstream
    tool_sources = get_mcp_tool_sources()

    # Use specific categories with limits
    tool_sources = get_mcp_tool_sources(categories=["Finance"], limit=2)

    # Use in evaluation
    solver = react(tools=tool_sources)
"""

import copy
import json
import logging
from typing import Dict, List, Optional, Any, Union, Literal
from inspect_ai.tool import mcp_tools, mcp_server_stdio, ToolSource

from .data_fetcher import get_tools_data, get_config_data

logger = logging.getLogger(__name__)


class MCPToolsRegistry:
    """
    Generic registry for MCP tools using inspect.ai's native MCP support.

    This class loads MCP server configurations and tool definitions from
    the upstream LiveMCPBench repository and creates ToolSource objects that
    can be directly used with inspect.ai's react solver.

    Always fetches data from the upstream repository for the latest tool definitions.
    """

    def __init__(self) -> None:
        """
        Initialize the MCP tools registry.

        Always fetches data from upstream LiveMCPBench repository.
        """
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
        ]

        # Load configurations from upstream
        self._load_configurations()
        self._categorize_tools()

    def _load_configurations(self) -> None:
        """Load tool and server configurations from upstream repository."""
        try:
            # Always use dynamic fetching from upstream
            raw_tools_data = get_tools_data()

            # Validate and filter tools data - exclude tools with missing descriptions
            self._tools_data = []
            excluded_count = 0
            for tool_data in raw_tools_data:
                validated_tool_data = self._validate_and_filter_tool_data(tool_data)
                if validated_tool_data is not None:
                    self._tools_data.append(validated_tool_data)
                else:
                    excluded_count += 1

            logger.info(
                f"Loaded {len(self._tools_data)} tool definitions from upstream repository "
                f"({excluded_count} tools excluded due to missing descriptions)"
            )

            self._server_configs = get_config_data()
            logger.info(
                f"Loaded {len(self._server_configs)} server configurations from upstream repository"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON data received from upstream: {e}")
            raise ValueError(f"Failed to parse upstream data: {e}")
        except Exception as e:
            logger.error(f"Error loading configurations from upstream: {e}")
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

    def _validate_and_filter_tool_data(
        self, tool_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Validate tool data and exclude tools/servers with missing descriptions.

        Args:
            tool_data: Raw tool data from upstream

        Returns:
            Tool data if valid, None if it should be excluded
        """
        # Create a deep copy to avoid modifying the original
        validated_data = copy.deepcopy(tool_data)

        # Check for missing or empty description at the top level
        if (
            "description" not in validated_data
            or not validated_data["description"]
            or validated_data["description"].strip() == ""
        ):
            logger.warning(
                f"Excluding tool '{validated_data.get('name', 'unknown')}' - missing top-level description"
            )
            return None

        # Validate tools field structure if it exists
        if "tools" in validated_data:
            servers_to_remove = []

            for server_name, server_tools in validated_data["tools"].items():
                if isinstance(server_tools, dict) and "tools" in server_tools:
                    has_invalid_tools = False

                    for tool in server_tools["tools"]:
                        if isinstance(tool, dict):
                            # Check for missing tool descriptions
                            if (
                                "description" not in tool
                                or not tool["description"]
                                or tool["description"].strip() == ""
                            ):
                                tool_name = tool.get("name", "unknown")
                                logger.warning(
                                    f"Excluding server '{server_name}' - tool '{tool_name}' has missing description"
                                )
                                has_invalid_tools = True
                                break

                    if has_invalid_tools:
                        servers_to_remove.append(server_name)

            # Remove servers with invalid tools
            for server_name in servers_to_remove:
                del validated_data["tools"][server_name]

            # If no valid servers remain, exclude the entire tool
            if not validated_data.get("tools"):
                logger.warning(
                    f"Excluding tool '{validated_data.get('name', 'unknown')}' - no valid servers remaining"
                )
                return None

        return validated_data

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
                        logger.debug(
                            f"Skipping server '{server_name}' - known to have issues"
                        )
                        continue

                    try:
                        # Validate server configuration before creating server
                        if (
                            "command" not in server_config
                            or not server_config["command"]
                        ):
                            logger.warning(
                                f"Server '{server_name}' has no command - skipping"
                            )
                            continue

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
        server_configs: Dict[str, Dict[str, Any]] = {}

        # Get config from the tool spec
        config = tool_spec.get("config", {})
        if not isinstance(config, dict):
            logger.warning(
                f"Tool '{tool_spec.get('name', 'unknown')}' has invalid config - skipping"
            )
            return server_configs

        mcp_servers = config.get("mcpServers", {})
        if not isinstance(mcp_servers, dict):
            logger.warning(
                f"Tool '{tool_spec.get('name', 'unknown')}' has invalid mcpServers config - skipping"
            )
            return server_configs

        for server_name, server_config in mcp_servers.items():
            if not isinstance(server_config, dict):
                logger.warning(
                    f"Server '{server_name}' in tool '{tool_spec.get('name', 'unknown')}' has invalid config - skipping"
                )
                continue
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
            if (
                not isinstance(server_tools_data, dict)
                or "tools" not in server_tools_data
            ):
                logger.warning(
                    f"Server '{server_name}' in tool '{tool_spec.get('name', 'unknown')}' has invalid tools structure, using 'all'"
                )
                return "all"

            try:
                # Extract tool names
                available_tools = []
                for tool in server_tools_data.get("tools", []):
                    if isinstance(tool, dict) and "name" in tool and tool["name"]:
                        available_tools.append(tool["name"])
                return available_tools
            except (KeyError, TypeError) as e:
                logger.warning(
                    f"Error extracting tool names from server '{server_name}': {e}, using 'all'"
                )
                return "all"

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
    categories: Optional[List[str]] = None,
    limit: Optional[int] = None,
    tools_per_server: Union[str, List[str]] = "all",
) -> List[ToolSource]:
    """
    Get MCP tool sources for use in inspect.ai evaluations using native MCP support.

    Args:
        categories: List of tool categories to include (e.g., ["Finance", "Discovery"])
        limit: Maximum number of servers per category to load (reduces installation logs)
        tools_per_server: Either "all" or list of specific tool names to include per server

    Returns:
        List of ToolSource objects ready for use in inspect.ai evaluations

    Example:
        # Use specific categories with limits
        tool_sources = get_mcp_tool_sources(categories=["Finance"], limit=2)

        # Use in a task
        solver = react(tools=tool_sources)
    """
    # Create registry and get tool sources
    registry = MCPToolsRegistry()

    return registry.create_tool_sources(
        categories=categories, limit=limit, tools_per_server=tools_per_server
    )


def get_tool_categories() -> List[str]:
    """Get all available tool categories from the upstream repository."""
    registry = MCPToolsRegistry()
    return registry.get_categories()


def get_registry_stats() -> Dict[str, Any]:
    """Get statistics about the tool registry from the upstream repository."""
    registry = MCPToolsRegistry()
    return registry.get_stats()
