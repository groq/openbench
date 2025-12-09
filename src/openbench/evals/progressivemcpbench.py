"""
ProgressiveMCPBench evaluation.

This benchmark evaluates LLM agents on real-world tasks using MCP tools.
It supports multiple strategies for tool discovery and execution.

Strategies:
- copilot: Uses semantic search with embeddings to discover tools via route/execute-tool
- directory: Presents tools as a filesystem with ls/read-tool-file/execute-tool
- minimal-servers: Provides only the tools from the required server(s) for each task
- minimal-tools: Provides only the exact tools needed for each task
- distraction-64: Minimal tools plus distraction tools to total 64 tools
- distraction-128: Minimal tools plus distraction tools to total 128 tools
- remote: Uses remote MCP servers via Groq's native MCP support (single-shot, no agentic loop)
"""

from inspect_ai import task, Task
from inspect_ai.solver import solver, Solver
from inspect_ai.agent import react, AgentPrompt
from inspect_ai.solver._task_state import TaskState
from inspect_ai.model import GenerateConfig, ChatMessageSystem
from inspect_ai.solver import Generate
from inspect_ai.tool import ToolError, ToolSource
import asyncio
import json
from pathlib import Path
from typing import Any

from openbench.datasets.progressivemcpbench import get_dataset
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
    "minimal-tools",
    "distraction-64",
    "distraction-128",
    "remote",
}


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


# Remote MCP configuration
REMOTE_MCP_BASE_URL = "https://progressive-mcp-bench.groq-dev.workers.dev/mcp"


def _synthetic_mcp_dir() -> Path:
    """Get the synthetic_mcp directory path."""
    current = Path(__file__).resolve()
    repo_root = current.parent.parent.parent.parent
    return repo_root / "synthetic_mcp"


def _load_servers_config() -> dict[str, Any]:
    """Load the synthetic servers.json configuration."""
    config_path = _synthetic_mcp_dir() / "config" / "servers.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _build_remote_mcp_tools(required_servers: list[str]) -> list[dict[str, Any]]:
    """Build remote MCP tool definitions for the specified servers.

    Each server becomes a single MCP tool entry that Groq's API will use
    to discover and execute tools from the remote server.

    Args:
        required_servers: List of server names to include

    Returns:
        List of MCP tool definitions for use in extra_body
    """
    servers_config = _load_servers_config()

    mcp_tools: list[dict[str, Any]] = []
    for server_name in required_servers:
        if server_name not in servers_config:
            continue

        server = servers_config[server_name]
        mcp_tool = {
            "type": "mcp",
            "server_label": server_name,
            "server_url": f"{REMOTE_MCP_BASE_URL}/{server_name}",
            "server_description": server.get(
                "description", f"MCP server: {server_name}"
            ),
            "require_approval": "never",
        }
        mcp_tools.append(mcp_tool)

    return mcp_tools


PROGRESSIVEMCPBENCH_REMOTE_SYSTEM_MESSAGE = """
You are an agent designed to assist users with daily tasks by using external tools.

You have access to remote MCP servers that provide various tools. The API will automatically
discover the available tools from each server and execute them on your behalf.

CRITICAL OUTPUT RULES:
- Output ONLY: {"final_answer": "your answer here"}
- DO NOT include tool_calls, reasoning, or any other fields
- DO NOT include complex data structures
- If you cannot determine an answer, return: {"final_answer": "I could not determine an answer"}
- Do not wrap the JSON in backticks or any other formatting
- The final_answer should be a concise string directly answering the user's question

Example final output:

{
  "final_answer": "The file contains 42 lines of text."
}

Output only this JSON object as your final response.
""".strip()


@solver
def progressive_remote_solver() -> Solver:
    """Solver that uses remote MCP servers via Groq's native MCP support.

    This strategy uses single-shot generation where Groq's API handles
    tool discovery and execution via remote MCP servers.
    Only works with minimal-servers configuration and Groq models.

    The solver passes MCP tools to the GroqMCPAPI provider via extra_body,
    which injects them into the request's tools array.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata or {}
        required_servers = metadata.get("required_servers", [])

        if not required_servers:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "missing_annotation"
            state.metadata["error_message"] = (
                "Task is missing 'required_servers' annotation."
            )
            return state

        mcp_tools = _build_remote_mcp_tools(required_servers)

        if not mcp_tools:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "no_valid_servers"
            state.metadata["error_message"] = (
                f"No valid MCP servers found for: {required_servers}"
            )
            return state

        state.messages.insert(
            0, ChatMessageSystem(content=PROGRESSIVEMCPBENCH_REMOTE_SYSTEM_MESSAGE)
        )

        try:
            state = await generate(
                state,
                tool_calls="none",
                extra_body={
                    "mcp_tools": mcp_tools,
                    "mcp_tool_choice": "auto",
                },
            )
        except Exception as e:
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "runtime_error"
            state.metadata["error_message"] = str(e)

        return state

    return solve


def _get_solver_for_strategy(strategy: str) -> Solver:
    """Get the appropriate solver for the given strategy."""
    if strategy == "copilot":
        return progressive_copilot_solver()
    elif strategy == "directory":
        return progressive_directory_solver()
    elif strategy == "minimal-servers":
        return progressive_minimal_servers_solver()
    elif strategy == "minimal-tools":
        return progressive_minimal_tools_solver()
    elif strategy == "distraction-64":
        return progressive_distraction_64_solver()
    elif strategy == "distraction-128":
        return progressive_distraction_128_solver()
    elif strategy == "remote":
        return progressive_remote_solver()
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Valid strategies: {VALID_STRATEGIES}"
        )


@task
def progressivemcpbench(
    working_limit: int = 60,
    strategy: str | None = None,
) -> Task:
    """ProgressiveMCPBench using configurable tool discovery strategies.

    All strategies now use the synthetic MCP layer with deterministic responses.

    Args:
        working_limit: Maximum number of API calls per task.
        strategy: Tool discovery strategy. Required. One of:
            - "copilot": Semantic search with embeddings (route/execute-tool)
            - "directory": Filesystem-like exploration (ls/read-tool-file/execute-tool)
            - "minimal-servers": Direct access to required server tools (requires annotations)
            - "minimal-tools": Direct access to exact required tools (requires annotations)
            - "distraction-64": Minimal tools + distractors to 64 total (requires annotations)
            - "distraction-128": Minimal tools + distractors to 128 total (requires annotations)
            - "remote": Uses Groq's remote MCP support (single-shot, no local server needed)
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

    if strategy != "remote":
        ensure_mcp_server_running()

    solver = _get_solver_for_strategy(strategy)

    # All strategies now use the synthetic dataset
    dataset = get_dataset()

    return Task(
        dataset=dataset,
        solver=[solver],
        scorer=progressivemcpbench_scorer(),
        name=f"progressivemcpbench-{strategy}",
        config=GenerateConfig(
            temperature=0.7,
            max_tokens=2048,
            # extra_body={"response_format": {"type": "json_object"}}, # removed to avoid 'json' tool issues
        ),
        working_limit=working_limit,
    )
