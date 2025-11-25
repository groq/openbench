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
from openbench.utils.text import progressivemcpbench_system_message


SUPPORTED_STRATEGIES = {"copilot", "directory"}


def _resolve_strategy(strategy: str | None) -> str:
    normalized = (strategy or "").strip().lower()
    if not normalized:
        raise ValueError(
            "ProgressiveMCPBench requires a strategy (-T strategy=copilot|directory)."
        )
    if normalized not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported strategy '{strategy}'. Choose one of: {', '.join(sorted(SUPPORTED_STRATEGIES))}."
        )
    return normalized


def _tool_sources_for_strategy(strategy: str):
    if strategy == "copilot":
        ts = copilot_tool_source()
        return [ts]
    if strategy == "directory":
        ts = directory_tool_source()
        return [ts]
    raise ValueError(f"Unsupported strategy '{strategy}'")


@solver
def progressive_solver(strategy: str) -> Solver:
    """Solver that routes to a specific ProgressiveMCPBench strategy."""
    resolved_strategy = _resolve_strategy(strategy)
    instructions = progressivemcpbench_system_message(resolved_strategy)
    tool_sources = _tool_sources_for_strategy(resolved_strategy)

    async def solve(state: TaskState, generate) -> TaskState:
        try:
            react_solver = react(
                prompt=AgentPrompt(
                    instructions=instructions,
                    assistant_prompt=None,
                    handoff_prompt=None,
                ),
                tools=tool_sources,
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


@task
def progressivemcpbench(
    strategy: str,
    working_limit: int = 600,
) -> Task:
    """ProgressiveMCPBench with selectable tool discovery strategies."""

    resolved_strategy = _resolve_strategy(strategy)

    return Task(
        dataset=get_dataset(),
        solver=[progressive_solver(resolved_strategy)],
        scorer=progressivemcpbench_scorer(),
        name="progressivemcpbench",
        metadata={"strategy": resolved_strategy},
        config=GenerateConfig(
            temperature=0.7,
            max_tokens=2048,
            extra_body={"response_format": {"type": "json_object"}},
        ),
        working_limit=working_limit,
    )
