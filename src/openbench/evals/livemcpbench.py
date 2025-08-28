from inspect_ai import task, Task
from inspect_ai.agent import react
from inspect_ai.model import GenerateConfig
from openbench.datasets.livemcpbench import get_dataset
from openbench.scorers.livemcpbench import livemcpbench_scorer
from openbench.tools.livemcpbench import get_mcp_tool_sources


@task
def livemcpbench(grader_model: str = "groq/llama-3.3-70b-versatile") -> Task:
    """LiveMCPBench: Evaluating LLM agents on real-world MCP tasks.

    This benchmark evaluates language model agents on their ability to complete
    real-world tasks using the Model Context Protocol (MCP). Tasks span multiple
    categories and require effective use of various MCP tools and servers.

    Args:
        grader_model: Model to use for grading responses (defaults to llama-3.1-8b-instruct)

    Returns:
        Task configured for LiveMCPBench evaluation
    """
    # Get MCP tool sources
    # NOTE: This will work for some models, but some hallucinate the tool names and cause 400s.
    # Try GPT-4o vs llama-3.3-70b-versatile. There are also issues setting up all 70 MCP servers from LiveMCPBench.
    # To see issues, get rid of categories and limit and run livemcpbench.
    mcp_tool_sources = get_mcp_tool_sources(
        categories=["Finance", "Discovery"], limit=3
    )

    return Task(
        dataset=get_dataset(),
        solver=[
            react(  # type: ignore[list-item]
                tools=mcp_tool_sources,
            )
        ],
        scorer=livemcpbench_scorer(model=grader_model),
        name="livemcpbench",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=6144,
        ),
    )
