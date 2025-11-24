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
from openbench.utils.text import PROGRESSIVEMCPBENCH_SYSTEM_MESSAGE


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
                    submit_prompt="",  # Suppress default submit instruction
                ),
                tools=[tool_source],
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
    working_limit: int = 600,
) -> Task:
    """ProgressiveMCPBench using the Copilot agent with JSON-structured output."""

    return Task(
        dataset=get_dataset(),
        solver=[progressive_copilot_solver()],
        scorer=progressivemcpbench_scorer(),
        name="progressivemcpbench",
        config=GenerateConfig(
            temperature=0.7,
            max_tokens=2048,
        ),
        working_limit=working_limit,
    )
