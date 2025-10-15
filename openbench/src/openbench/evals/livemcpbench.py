from inspect_ai import task, Task
from inspect_ai.solver import solver, Solver
from inspect_ai.agent import react, AgentPrompt
from inspect_ai.solver._task_state import TaskState
from inspect_ai.model import GenerateConfig
from inspect_ai.tool import ToolError
import asyncio
from openbench.datasets.livemcpbench import get_dataset
from openbench.scorers.livemcpbench import livemcpbench_scorer
from openbench.tools.livemcpbench.copilot.toolsource import copilot_tool_source
from openbench.utils.text import LIVEMCPBENCH_SYSTEM_MESSAGE


@solver
def copilot_solver() -> Solver:
    """Solver that uses the Copilot MCP server."""

    async def solve(state: TaskState, generate) -> TaskState:
        try:
            tool_source = copilot_tool_source()
            react_solver = react(
                prompt=AgentPrompt(
                    instructions=LIVEMCPBENCH_SYSTEM_MESSAGE,
                    assistant_prompt=None,
                    handoff_prompt=None,
                    submit_prompt=None,
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
def livemcpbench(
    grader_model: str = "openai/gpt-4.1-mini-2025-04-14",
    working_limit: int = 600,
) -> Task:
    """LiveMCPBench using the baseline Copilot agent (route + execute-tool)."""

    return Task(
        dataset=get_dataset(),
        solver=[copilot_solver()],
        scorer=livemcpbench_scorer(model=grader_model),
        name="livemcpbench",
        config=GenerateConfig(
            temperature=0.7,
            max_tokens=2048,
        ),
        working_limit=working_limit,
    )
