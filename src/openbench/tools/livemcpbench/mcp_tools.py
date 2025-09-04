"""
MCP Tools Integration for OpenBench with Semantic Search

This module provides a framework for integrating MCP (Model Context Protocol)
tools into OpenBench evaluations using semantic search to find the most
relevant tools for each task.

Usage:
    from openbench.tools.livemcpbench import MCPToolsRegistry

    # Create registry and initialize semantic retriever
    registry = MCPToolsRegistry()
    registry.init_retriever()

    # Get tools for a specific task using semantic search
    tools = registry.create_tool_sources_semantic(
        query="Generate a PDF report with charts",
        top_k_servers=5,
        top_k_tools_per_server=3
    )
"""

import copy
import json
import logging
import os
from typing import Dict, List, Optional, Any
from inspect_ai.tool import mcp_tools, mcp_server_stdio, ToolSource

from .data_fetcher import get_tools_data, get_config_data
from .tool_retriever import EmbeddingToolRetriever

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

        self._problem_servers: List[str] = [
            "text-editor",  # has known schema validation issues
            "searxng",  # File Access category - prone to validation issues
            "okppt",  # requires pycairo which needs Cairo graphics library
            "chess",  # requires cairosvg/cairocffi which can't find Cairo in uvx environment
        ]

        # Embedding retriever
        self._retriever: Optional[EmbeddingToolRetriever] = None

        # Load configurations from upstream
        self._load_configurations()

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

    def init_retriever(
        self,
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize the embedding-based retriever."""
        self._retriever = EmbeddingToolRetriever(
            embedding_model=embedding_model, api_key=api_key, base_url=base_url
        )
        # Index all tools
        self._retriever.index_tools(self._tools_data, self._server_configs)
        logger.info("Initialized embedding-based tool retriever")

    def create_tool_sources_semantic(
        self,
        query: str,
        top_k_servers: int = 6,
        top_k_tools_per_server: int = 4,
        category_hint: Optional[str] = None,
        suppress_server_output: bool = True,
    ) -> List[ToolSource]:
        """
        Create tool sources using semantic search.

        Args:
            query: Task description or query
            top_k_servers: Number of top servers to retrieve
            top_k_tools_per_server: Max tools per server
            category_hint: Optional category to boost relevance
            suppress_server_output: If True, suppress server startup output

        Returns:
            List of ToolSource objects ready for use with inspect.ai
        """
        if not self._retriever:
            logger.error("Retriever not initialized. Call init_retriever() first.")
            return []

        # Get relevant tools
        relevant_tools = self._retriever.retrieve_tools(
            query, top_k_servers, top_k_tools_per_server, category_hint
        )

        tool_sources = []
        processed_servers = set()

        for server_name, tool_names in relevant_tools:
            if server_name in processed_servers or server_name in self._problem_servers:
                continue

            # Find the server config
            server_config = None
            for tool_spec in self._tools_data:
                config = tool_spec.get("config", {})
                mcp_servers = config.get("mcpServers", {})
                if server_name in mcp_servers:
                    server_config = mcp_servers[server_name]
                    break

            if not server_config or "command" not in server_config:
                continue

            try:
                # Create MCP server
                server_env = (
                    server_config.get("env", {}).copy()
                    if server_config.get("env")
                    else {}
                )

                # Suppress server output if requested
                if suppress_server_output:
                    # Set environment variables to suppress output
                    server_env.update(
                        {
                            "NODE_NO_WARNINGS": "1",  # Suppress Node.js warnings
                            "PYTHONWARNINGS": "ignore",  # Suppress Python warnings
                            "SUPPRESS_OUTPUT": "1",  # Generic flag for tools that support it
                        }
                    )

                    # Try to detect if we can use output redirection
                    # Note: This is a workaround since mcp_server_stdio might not expose stdio control
                    if os.name != "nt":  # Unix-like systems
                        # Set minimal logging for common tools
                        server_env.update(
                            {
                                "LOG_LEVEL": "error",
                                "RUST_LOG": "error",
                                "DEBUG": "0",
                            }
                        )

                mcp_server = mcp_server_stdio(
                    command=server_config["command"],
                    args=server_config.get("args", []),
                    cwd=server_config.get("cwd"),
                    env=server_env,
                )

                # Create tool source with specific tools
                tool_source = mcp_tools(mcp_server, tools=tool_names)
                tool_sources.append(tool_source)
                processed_servers.add(server_name)

                logger.info(
                    f"Created semantic tool source for '{server_name}' with tools: {tool_names}"
                )

            except Exception as e:
                logger.warning(f"Failed to create tool source for '{server_name}': {e}")
                continue

        logger.info(f"Created {len(tool_sources)} tool sources via semantic search")
        return tool_sources

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the tool registry."""
        return {
            "total_tools": len(self._tools_data),
            "total_servers": len(self._server_configs),
            "retriever_initialized": self._retriever is not None,
        }


def get_registry_stats() -> Dict[str, Any]:
    """Get statistics about the tool registry from the upstream repository."""
    registry = MCPToolsRegistry()
    return registry.get_stats()
