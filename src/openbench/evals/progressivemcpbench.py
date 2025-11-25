"""
ProgressiveMCPBench evaluation.

This benchmark evaluates LLM agents on real-world tasks using MCP tools.
It supports multiple strategies for tool discovery and execution.

Strategies:
- copilot: Uses semantic search with embeddings to discover tools via route/execute-tool
- directory: Presents tools as a filesystem with ls/read-tool-file/execute-tool
- minimal-servers: Provides only the tools from the required server(s) for each task
- minimal-tools: Provides only the exact tools needed for each task
- distraction-128: Provides required tools plus distraction tools to total 128 tools
"""

from inspect_ai import task, Task
from inspect_ai.solver import solver, Solver
from inspect_ai.agent import react, AgentPrompt
from inspect_ai.solver._task_state import TaskState
from inspect_ai.model import GenerateConfig
from inspect_ai.tool import ToolError
import asyncio

from openbench.datasets.progressivemcpbench import get_dataset
from openbench.scorers.progressivemcpbench import progressivemcpbench_scorer
from openbench.tools.progressivemcpbench.copilot.toolsource import copilot_tool_source
from openbench.tools.progressivemcpbench.directory.toolsource import (
    directory_tool_source,
)
from openbench.utils.text import (
    PROGRESSIVEMCPBENCH_SYSTEM_MESSAGE,
    PROGRESSIVEMCPBENCH_DIRECTORY_SYSTEM_MESSAGE,
    PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE,
)


VALID_STRATEGIES = {
    "copilot",
    "directory",
    "minimal-servers",
    "minimal-tools",
    "distraction-128",
}


def _get_system_message(strategy: str) -> str:
    """Get the appropriate system message for the given strategy."""
    if strategy == "copilot":
        return PROGRESSIVEMCPBENCH_SYSTEM_MESSAGE
    elif strategy == "directory":
        return PROGRESSIVEMCPBENCH_DIRECTORY_SYSTEM_MESSAGE
    else:
        return PROGRESSIVEMCPBENCH_MINIMAL_SYSTEM_MESSAGE


@solver
def progressive_copilot_solver() -> Solver:
    """Solver that uses the Copilot MCP server for ProgressiveMCPBench."""

    async def solve(state: TaskState, generate) -> TaskState:
        try:
            tool_source = copilot_tool_source()
            react_solver = react(
                prompt=AgentPrompt(
                    instructions=PROGRESSIVEMCPBENCH_SYSTEM_MESSAGE,
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

    return solve


@solver
def progressive_directory_solver() -> Solver:
    """Solver that uses the Directory MCP server for ProgressiveMCPBench."""

    async def solve(state: TaskState, generate) -> TaskState:
        try:
            tool_source = directory_tool_source()
            react_solver = react(
                prompt=AgentPrompt(
                    instructions=PROGRESSIVEMCPBENCH_DIRECTORY_SYSTEM_MESSAGE,
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

    return solve


def _get_solver_for_strategy(strategy: str) -> Solver:
    """Get the appropriate solver for the given strategy."""
    if strategy == "copilot":
        return progressive_copilot_solver()
    elif strategy == "directory":
        return progressive_directory_solver()
    elif strategy in {"minimal-servers", "minimal-tools", "distraction-128"}:
        raise NotImplementedError(
            f"Strategy '{strategy}' requires task-specific tool annotations. "
            "Please run the benchmark with strategy=copilot or strategy=directory first "
            "to generate success logs, then annotate the dataset."
        )
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Valid strategies: {VALID_STRATEGIES}"
        )


@task
def progressivemcpbench(
    working_limit: int = 600,
    strategy: str | None = None,
) -> Task:
    """ProgressiveMCPBench using configurable tool discovery strategies.

    Args:
        working_limit: Maximum number of API calls per task.
        strategy: Tool discovery strategy. Required. One of:
            - "copilot": Semantic search with embeddings (route/execute-tool)
            - "directory": Filesystem-like exploration (ls/read-tool-file/execute-tool)
            - "minimal-servers": Direct access to required server tools (requires annotations)
            - "minimal-tools": Direct access to exact required tools (requires annotations)
            - "distraction-128": Required tools + distractors to 128 total (requires annotations)
    """
    if strategy is None:
        raise ValueError(
            "The 'strategy' task variable is required for ProgressiveMCPBench.\n"
            "Use -T strategy=copilot or -T strategy=directory on the command line.\n"
            f"Valid strategies: {', '.join(sorted(VALID_STRATEGIES))}"
        )

    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}'.\n"
            f"Valid strategies: {', '.join(sorted(VALID_STRATEGIES))}"
        )

    solver = _get_solver_for_strategy(strategy)

    return Task(
        dataset=get_dataset(),
        solver=[solver],
        scorer=progressivemcpbench_scorer(),
        name=f"progressivemcpbench-{strategy}",
        config=GenerateConfig(
            temperature=0.7,
            max_tokens=2048,
            extra_body={"response_format": {"type": "json_object"}},
        ),
        working_limit=working_limit,
    )
