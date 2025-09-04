"""
Embedding-based tool retrieval for LiveMCPBench.

This module implements semantic search for MCP tools using embeddings,
allowing selection of the most relevant tools based on task descriptions.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from .embedding_cache import get_embedding_cache

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """Information about a tool for retrieval."""

    server_name: str
    tool_name: str
    description: str
    category: str
    full_spec: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class EmbeddingToolRetriever:
    """
    Retrieves relevant MCP tools using embedding-based semantic search.

    This class computes embeddings for tool descriptions and uses cosine
    similarity to find the most relevant tools for a given task.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        cache_embeddings: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the embedding-based tool retriever.

        Args:
            embedding_model: OpenAI embedding model to use
            embedding_dim: Dimension of embeddings
            cache_embeddings: Whether to cache computed embeddings
            api_key: OpenAI API key (uses env var if not provided)
            base_url: Optional base URL for API
        """
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.cache_embeddings = cache_embeddings

        # Set up OpenAI client
        self.client: Optional[Any] = None
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url
            )
        except ImportError:
            logger.warning(
                "OpenAI package not installed. Install with: pip install openai"
            )

        self.tool_embeddings: Dict[str, ToolInfo] = {}

        # Use persistent cache
        self._cache = get_embedding_cache() if cache_embeddings else None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string."""
        if not self.client:
            # Fallback: simple random embedding for testing
            logger.warning("No OpenAI client available, using random embeddings")
            return np.random.randn(self.embedding_dim)

        # Check persistent cache first
        if self._cache:
            cached = self._cache.get(text, self.embedding_model)
            if cached is not None:
                return cached

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model, input=text
            )
            embedding = np.array(response.data[0].embedding)

            # Save to persistent cache
            if self._cache:
                self._cache.set(text, self.embedding_model, embedding)

            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return random embedding as fallback
            return np.random.randn(self.embedding_dim)

    def index_tools(
        self, tools_data: List[Dict[str, Any]], server_configs: List[Dict[str, Any]]
    ) -> None:
        """
        Index tools by computing embeddings for their descriptions.

        Args:
            tools_data: List of tool specifications from LiveMCPBench
            server_configs: List of server configurations
        """
        logger.info(f"Indexing {len(tools_data)} tool specifications...")

        for tool_spec in tools_data:
            category = tool_spec.get("category", "Miscellaneous")

            # Extract tools from each server in the spec
            tools_dict = tool_spec.get("tools", {})
            for server_name, server_tools in tools_dict.items():
                if not isinstance(server_tools, dict) or "tools" not in server_tools:
                    continue

                for tool in server_tools.get("tools", []):
                    if not isinstance(tool, dict):
                        continue

                    tool_name = tool.get("name", "")
                    description = tool.get("description", "")

                    if not tool_name or not description:
                        continue

                    # Create combined text for embedding
                    # Include category and tool name in the embedding
                    embedding_text = f"{category}: {tool_name} - {description}"

                    # Create unique key
                    tool_key = f"{server_name}::{tool_name}"

                    # Create tool info
                    tool_info = ToolInfo(
                        server_name=server_name,
                        tool_name=tool_name,
                        description=description,
                        category=category,
                        full_spec=tool_spec,
                        embedding=self._get_embedding(embedding_text),
                    )

                    self.tool_embeddings[tool_key] = tool_info

        logger.info(f"Indexed {len(self.tool_embeddings)} tools with embeddings")

    def retrieve_tools(
        self,
        query: str,
        top_k_servers: int = 5,
        top_k_tools_per_server: int = 3,
        category_hint: Optional[str] = None,
    ) -> List[Tuple[str, List[str]]]:
        """
        Retrieve relevant tools based on semantic similarity to query.

        Args:
            query: Task description or query
            top_k_servers: Number of top servers to return
            top_k_tools_per_server: Max tools per server
            category_hint: Optional category to boost relevance

        Returns:
            List of (server_name, [tool_names]) tuples
        """
        if not self.tool_embeddings:
            logger.warning("No tools indexed. Call index_tools() first.")
            return []

        # Get query embedding
        query_text = query
        if category_hint:
            query_text = f"{category_hint}: {query}"
        query_embedding = self._get_embedding(query_text)

        # Compute similarities
        similarities = []
        for tool_key, tool_info in self.tool_embeddings.items():
            if tool_info.embedding is None:
                continue

            # Cosine similarity
            similarity = np.dot(query_embedding, tool_info.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(tool_info.embedding)
            )

            # Boost similarity if category matches hint
            if category_hint and tool_info.category.lower() == category_hint.lower():
                similarity *= 1.2  # 20% boost for matching category

            similarities.append((similarity, tool_info))

        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Group by server and select top tools
        server_tools: Dict[str, List[Tuple[float, str]]] = {}
        for similarity, tool_info in similarities:
            server_name = tool_info.server_name

            if server_name not in server_tools:
                server_tools[server_name] = []

            if len(server_tools[server_name]) < top_k_tools_per_server:
                server_tools[server_name].append((similarity, tool_info.tool_name))

        # Select top-k servers based on their best tool's similarity
        server_scores = []
        for server_name, tools in server_tools.items():
            if tools:
                best_score = max(score for score, _ in tools)
                server_scores.append((best_score, server_name))

        server_scores.sort(key=lambda x: x[0], reverse=True)

        # Return top servers with their tools
        result = []
        for _, server_name in server_scores[:top_k_servers]:
            tool_names = [
                name for _, name in sorted(server_tools[server_name], reverse=True)
            ]
            result.append((server_name, tool_names))

        return result

    def get_relevant_tool_specs(
        self,
        query: str,
        top_k_servers: int = 5,
        top_k_tools_per_server: int = 3,
        category_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get full tool specifications for relevant tools.

        Returns the tool specs needed to create MCP servers.
        """
        relevant_tools = self.retrieve_tools(
            query, top_k_servers, top_k_tools_per_server, category_hint
        )

        # Collect unique tool specs
        seen_specs = set()
        tool_specs = []

        for server_name, tool_names in relevant_tools:
            # Find the tool spec containing this server
            for tool_info in self.tool_embeddings.values():
                if (
                    tool_info.server_name == server_name
                    and id(tool_info.full_spec) not in seen_specs
                ):
                    seen_specs.add(id(tool_info.full_spec))
                    tool_specs.append(tool_info.full_spec)
                    break

        return tool_specs
