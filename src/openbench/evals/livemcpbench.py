import logging
from typing import Optional
from inspect_ai import task, Task
from inspect_ai.agent import react, agent, Agent
from inspect_ai.model import GenerateConfig
from openbench.datasets.livemcpbench import get_dataset
from openbench.scorers.livemcpbench import livemcpbench_scorer
from openbench.tools.livemcpbench import MCPToolsRegistry

logger = logging.getLogger(__name__)

# Global registry instance to avoid re-initialization
_global_registry = None


@agent
def MCP_copilot_agent(
    top_k_servers: int = 5,
    top_k_tools_per_server: int = 3,
    embedding_model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    suppress_server_output: bool = True,
) -> Agent:
    """
    MCP copilot agent from livemcpbench that uses semantic search to select the most relevant tools.

    Args:
        top_k_servers: Number of most relevant servers to retrieve
        top_k_tools_per_server: Max tools per server to include
        embedding_model: OpenAI embedding model to use
        api_key: Optional OpenAI API key
        base_url: Optional API base URL
        suppress_server_output: If True, suppress server startup messages

    Returns:
        Solver that applies react with semantically selected tools
    """
    # Use global registry to avoid re-initialization
    global _global_registry
    if _global_registry is None:
        _global_registry = MCPToolsRegistry()
        _global_registry.init_retriever(
            embedding_model=embedding_model, api_key=api_key, base_url=base_url
        )
        logger.info("Initialized global MCP tools registry with embeddings")

    registry = _global_registry

    # For now, we'll use a static set of tools based on common needs
    # In the future, this could be made dynamic per task
    tool_sources = registry.create_tool_sources_semantic(
        query="general purpose tools for file operations, web search, and data analysis",
        top_k_servers=top_k_servers,
        top_k_tools_per_server=top_k_tools_per_server,
        category_hint=None,
        suppress_server_output=suppress_server_output,
    )

    return react(tools=tool_sources)


@task
def livemcpbench(
    grader_model: str = "groq/llama-3.3-70b-versatile",
    top_k_servers: int = 5,
    top_k_tools_per_server: int = 3,
    embedding_model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    suppress_server_output: bool = True,
) -> Task:
    """LiveMCPBench: Evaluating LLM agents on real-world MCP tasks.

    This benchmark evaluates language model agents on their ability to complete
    real-world tasks using the Model Context Protocol (MCP). Uses embedding-based
    semantic search to find the most relevant tools for each task.

    Args:
        grader_model: Model to use for grading responses
        top_k_servers: Number of most relevant servers to retrieve
        top_k_tools_per_server: Max tools per server to include
        embedding_model: OpenAI embedding model for semantic search
        api_key: Optional OpenAI API key (uses env var if not provided)
        base_url: Optional API base URL
        suppress_server_output: If True, suppress server startup messages

    Returns:
        Task configured for LiveMCPBench evaluation with semantic retrieval
    """
    return Task(
        dataset=get_dataset(),
        solver=MCP_copilot_agent(
            top_k_servers=top_k_servers,
            top_k_tools_per_server=top_k_tools_per_server,
            embedding_model=embedding_model,
            api_key=api_key,
            base_url=base_url,
            suppress_server_output=suppress_server_output,
        ),
        scorer=livemcpbench_scorer(model=grader_model),
        name="livemcpbench",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=6144,
        ),
    )
