"""
ProgressiveMCPBench evaluation.

This benchmark evaluates LLM agents on real-world tasks using MCP tools.
It supports multiple strategies for tool discovery and execution.

Strategies:
- copilot: Uses semantic search with embeddings to discover tools via route/execute-tool
- directory: Presents tools as a filesystem with ls/read-tool-file/execute-tool
- minimal-servers: Provides only the tools from the required server(s) for each task
- minimal-servers-remote: Uses server-side MCP (Groq, Anthropic) with optional tool discovery
- minimal-tools: Provides only the exact tools needed for each task
- distraction-64: Minimal tools plus distraction tools to total 64 tools
- distraction-128: Minimal tools plus distraction tools to total 128 tools

Tool Discovery Options (for minimal-servers-remote):
- None: No advanced tool discovery
- directory: Groq's deferred_mode="directory" for tool discovery
- directory-lazy: Groq's deferred_mode="directory" with lazy loading
- regex: Anthropic's tool_search_tool_regex for regex-based search
- bm25: Anthropic's tool_search_tool_bm25 for BM25-based search
"""

from inspect_ai import task, Task
from inspect_ai.solver import solver, Solver
from inspect_ai.agent import react, AgentPrompt
from inspect_ai.solver._task_state import TaskState
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.tool import ToolError, ToolSource
import asyncio
import json
from typing import Any

from openbench.datasets.progressivemcpbench import get_dataset
from openbench.evals.remote_mcp import get_remote_mcp_handler
from openbench.scorers.progressivemcpbench import progressivemcpbench_scorer
from openbench.tools.progressivemcpbench.directory.toolsource import (
    directory_tool_source,
)
from openbench.tools.progressivemcpbench.synthetic.toolsource import (
    synthetic_minimal_servers_tool_source,
    synthetic_minimal_tools_tool_source,
    synthetic_distraction_64_tool_source,
    synthetic_distraction_128_tool_source,
)
from openbench.tools.progressivemcpbench.copilot.synthetic_toolsource import (
    synthetic_copilot_tool_source,
)
from openbench.utils.text import (
    PROGRESSIVEMCPBENCH_SYSTEM_MESSAGE,
    PROGRESSIVEMCPBENCH_DIRECTORY_SYSTEM_MESSAGE,
    PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE,
)
from openbench.utils.mcp_server_manager import ensure_mcp_server_running


VALID_STRATEGIES = {
    "copilot",
    "directory",
    "minimal-servers",
    "minimal-servers-remote",
    "minimal-tools",
    "distraction-64",
    "distraction-128",
}

VALID_TOOL_DISCOVERY_OPTIONS = {"directory", "directory-lazy", "regex", "bm25"}


def _get_system_message(strategy: str) -> str:
    """Get the appropriate system message for the given strategy."""
    if strategy == "copilot":
        return PROGRESSIVEMCPBENCH_SYSTEM_MESSAGE
    elif strategy == "directory":
        return PROGRESSIVEMCPBENCH_DIRECTORY_SYSTEM_MESSAGE
    else:
        return PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE


async def _run_react_with_tools(
    state: TaskState,
    system_message: str,
    tool_source: ToolSource,
) -> TaskState:
    """Run react solver with the given tools and handle errors."""
    try:
        react_solver = react(
            prompt=AgentPrompt(
                instructions=system_message,
                assistant_prompt=None,
                handoff_prompt=None,
            ),
            tools=[tool_source],
            submit=False,
        )
        return await react_solver(state)  # type: ignore[return-value, arg-type]
    except asyncio.TimeoutError:
        state.metadata = state.metadata or {}
        state.metadata["execution_error"] = "timeout"
        state.metadata["error_message"] = "Task execution timed out"
        return state
    except ToolError as e:
        state.metadata = state.metadata or {}
        state.metadata["execution_error"] = "tool_error"
        state.metadata["error_message"] = str(e)
        if state.output and not state.output.completion:
            state.output.completion = f"Task failed due to tool error: {str(e)}"
        return state
    except Exception as e:
        state.metadata = state.metadata or {}
        state.metadata["execution_error"] = "runtime_error"
        state.metadata["error_message"] = str(e)
        if state.output and not state.output.completion:
            state.output.completion = f"Task failed due to runtime error: {str(e)}"
        return state


@solver
def progressive_copilot_solver() -> Solver:
    """Solver that uses the synthetic Copilot MCP server for ProgressiveMCPBench.

    This strategy uses semantic search with embeddings to discover tools,
    then executes them via the synthetic HTTP MCP backend.
    """

    async def solve(state: TaskState, generate: Any) -> TaskState:
        tool_source = synthetic_copilot_tool_source()
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_SYSTEM_MESSAGE, tool_source
        )

    return solve


@solver
def progressive_directory_solver() -> Solver:
    """Solver that uses the Directory MCP server for ProgressiveMCPBench."""

    async def solve(state: TaskState, generate: Any) -> TaskState:
        tool_source = directory_tool_source()
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_DIRECTORY_SYSTEM_MESSAGE, tool_source
        )

    return solve


@solver
def progressive_minimal_servers_solver() -> Solver:
    """Solver that provides all tools from the required server(s) for each task."""

    async def solve(state: TaskState, generate: Any) -> TaskState:
        metadata = state.metadata or {}
        required_servers = metadata.get("required_servers", [])

        if not required_servers:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "missing_annotation"
            state.metadata["error_message"] = (
                "Task is missing 'required_servers' annotation."
            )
            return state

        tool_source = synthetic_minimal_servers_tool_source(required_servers)
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE, tool_source
        )

    return solve


@solver
def progressive_minimal_tools_solver() -> Solver:
    """Solver that provides only the exact tools needed for each task."""

    async def solve(state: TaskState, generate: Any) -> TaskState:
        metadata = state.metadata or {}
        required_servers = metadata.get("required_servers", [])
        required_tools_list = metadata.get("required_tools", [])

        if not required_servers or not required_tools_list:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "missing_annotation"
            state.metadata["error_message"] = (
                "Task is missing 'required_servers' or 'required_tools' annotation."
            )
            return state

        required_tools = _build_required_tools(required_servers, required_tools_list)
        tool_source = synthetic_minimal_tools_tool_source(required_tools)
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE, tool_source
        )

    return solve


def _build_required_tools(
    required_servers: list[str], required_tools_list: list[str]
) -> list[tuple[str, str]]:
    """Build list of (server, tool) tuples from annotations."""
    required_tools: list[tuple[str, str]] = []
    for server in required_servers:
        for tool in required_tools_list:
            required_tools.append((server, tool))
    return required_tools


@solver
def progressive_distraction_64_solver() -> Solver:
    """Solver that provides required tools plus distractors to total 64 tools."""

    async def solve(state: TaskState, generate: Any) -> TaskState:
        metadata = state.metadata or {}
        required_servers = metadata.get("required_servers", [])
        required_tools_list = metadata.get("required_tools", [])
        task_id = str(state.sample_id or "")

        if not required_servers or not required_tools_list:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "missing_annotation"
            state.metadata["error_message"] = (
                "Task is missing 'required_servers' or 'required_tools' annotation."
            )
            return state

        required_tools = _build_required_tools(required_servers, required_tools_list)
        tool_source = synthetic_distraction_64_tool_source(required_tools, task_id)
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE, tool_source
        )

    return solve


@solver
def progressive_distraction_128_solver() -> Solver:
    """Solver that provides required tools plus distractors to total 128 tools."""

    async def solve(state: TaskState, generate: Any) -> TaskState:
        metadata = state.metadata or {}
        required_servers = metadata.get("required_servers", [])
        required_tools_list = metadata.get("required_tools", [])
        task_id = str(state.sample_id or "")

        if not required_servers or not required_tools_list:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "missing_annotation"
            state.metadata["error_message"] = (
                "Task is missing 'required_servers' or 'required_tools' annotation."
            )
            return state

        required_tools = _build_required_tools(required_servers, required_tools_list)
        tool_source = synthetic_distraction_128_tool_source(required_tools, task_id)
        return await _run_react_with_tools(
            state, PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE, tool_source
        )

    return solve


def _load_servers_config() -> dict[str, Any]:
    """Load the synthetic servers.json configuration."""
    from openbench.utils.progressivemcpbench_cache import (
        get_progressivemcpbench_dataset_dir,
    )

    dataset_dir = get_progressivemcpbench_dataset_dir()
    config_path = dataset_dir / "config" / "servers.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


@solver
def _progressive_minimal_servers_remote_solver(
    tool_discovery: str | None = None,
) -> Solver:
    """Solver for minimal-servers-remote strategy using server-side MCP.

    This strategy:
    - Makes a single call to the provider's API with MCP server specs
    - Provider handles all tool discovery and execution internally
    - No local tool-calling loop (unlike other strategies)
    - Supports Groq (Responses API) and Anthropic (MCP connector)

    Args:
        tool_discovery: Optional tool discovery mode:
            - None: No advanced tool discovery
            - "directory": Groq's deferred_mode for directory-based discovery
            - "regex": Anthropic's tool_search_tool with regex patterns
            - "bm25": Anthropic's tool_search_tool with BM25 search
    """

    async def solve(state: TaskState, generate: Any) -> TaskState:
        metadata = state.metadata or {}
        required_servers = metadata.get("required_servers", [])

        if not required_servers:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "missing_annotation"
            state.metadata["error_message"] = (
                "Task is missing 'required_servers' annotation."
            )
            return state

        model = get_model()

        handler = get_remote_mcp_handler(model.api, tool_discovery)

        servers_config = _load_servers_config()
        system_message = _get_system_message("minimal-servers")

        return await handler.execute(
            state=state,
            required_servers=required_servers,
            servers_config=servers_config,
            system_message=system_message,
        )

    return solve


def _get_solver_for_strategy(
    strategy: str,
    tool_discovery: str | None = None,
) -> Solver:
    """Get the appropriate solver for the given strategy.

    Args:
        strategy: The strategy name
        tool_discovery: Optional tool discovery mode (only for minimal-servers-remote)
    """
    if strategy == "copilot":
        return progressive_copilot_solver()
    elif strategy == "directory":
        return progressive_directory_solver()
    elif strategy == "minimal-servers":
        return progressive_minimal_servers_solver()
    elif strategy == "minimal-servers-remote":
        return _progressive_minimal_servers_remote_solver(tool_discovery)
    elif strategy == "minimal-tools":
        return progressive_minimal_tools_solver()
    elif strategy == "distraction-64":
        return progressive_distraction_64_solver()
    elif strategy == "distraction-128":
        return progressive_distraction_128_solver()
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Valid strategies: {VALID_STRATEGIES}"
        )


@task
def progressivemcpbench(
    working_limit: int = 60,
    strategy: str | None = None,
    tool_discovery: str | None = None,
) -> Task:
    """ProgressiveMCPBench using configurable tool discovery strategies.

    All strategies now use the synthetic MCP layer with deterministic responses.

    Args:
        working_limit: Maximum number of API calls per task.
        strategy: Tool discovery strategy. Required. One of:
            - "copilot": Semantic search with embeddings (route/execute-tool)
            - "directory": Filesystem-like exploration (ls/read-tool-file/execute-tool)
            - "minimal-servers": Direct access to required server tools (requires annotations)
            - "minimal-servers-remote": Server-side MCP (Groq, Anthropic)
            - "minimal-tools": Direct access to exact required tools (requires annotations)
            - "distraction-64": Minimal tools + distractors to 64 total (requires annotations)
            - "distraction-128": Minimal tools + distractors to 128 total (requires annotations)
        tool_discovery: Optional tool discovery mode (only for minimal-servers-remote):
            - "directory": Groq's deferred_mode for directory-based tool discovery
            - "directory-lazy": Groq's deferred_mode with lazy loading
            - "regex": Anthropic's tool_search_tool with regex patterns
            - "bm25": Anthropic's tool_search_tool with BM25 search
    """
    if strategy is None:
        raise ValueError(
            "The 'strategy' task variable is required for ProgressiveMCPBench.\n"
            "Use -T strategy=directory or -T strategy=minimal-tools on the command line.\n"
            f"Valid strategies: {', '.join(sorted(VALID_STRATEGIES))}"
        )

    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}'.\n"
            f"Valid strategies: {', '.join(sorted(VALID_STRATEGIES))}"
        )

    if tool_discovery is not None:
        if strategy != "minimal-servers-remote":
            raise ValueError(
                f"tool_discovery='{tool_discovery}' is only valid with "
                f"strategy='minimal-servers-remote'. Got strategy='{strategy}'."
            )
        if tool_discovery not in VALID_TOOL_DISCOVERY_OPTIONS:
            raise ValueError(
                f"Invalid tool_discovery='{tool_discovery}'.\n"
                f"Valid options: {', '.join(sorted(VALID_TOOL_DISCOVERY_OPTIONS))}"
            )

    ensure_mcp_server_running()

    solver = _get_solver_for_strategy(strategy, tool_discovery)

    # All strategies now use the synthetic dataset
    dataset = get_dataset()

    # Build task name including tool_discovery if set
    task_name = f"progressivemcpbench-{strategy}"
    if tool_discovery:
        task_name = f"{task_name}-{tool_discovery}"

    return Task(
        dataset=dataset,
        solver=[solver],
        scorer=progressivemcpbench_scorer(),
        name=task_name,
        config=GenerateConfig(
            temperature=0.7,
            max_tokens=2048,
        ),
        working_limit=working_limit,
    )
