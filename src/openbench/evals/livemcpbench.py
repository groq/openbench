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
from openbench.utils.text import LIVEMPCBENCH_SYSTEM_MESSAGE
from pathlib import Path
import os


@solver
def copilot_solver() -> Solver:
    """Solver that uses the Copilot MCP server."""

    async def solve(state: TaskState, generate) -> TaskState:
        # Preflight: ensure embeddings file exists unless autogen is enabled
        mcp_data_path = os.getenv("MCP_DATA_PATH")
        if not mcp_data_path:
            embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            abstract_model = os.getenv("ABSTRACT_MODEL", "gpt-4.1-2025-04-14")
            cache_dir = Path(
                os.path.expanduser("~/.cache/openbench/livemcpbench/copilot/config")
            )
            mcp_data_path = str(
                cache_dir / f"mcp_arg_{embedding_model}_{abstract_model}.json"
            )
        if not os.path.exists(mcp_data_path) and os.getenv(
            "OPENBENCH_COPILOT_AUTOGEN", "0"
        ) not in {"1", "true", "True"}:
            msg = (
                "Copilot embeddings file not found: "
                + mcp_data_path
                + ". Run 'openbench mcp-copilot-prepare' first, set MCP_DATA_PATH to an existing file, "
                + "or set OPENBENCH_COPILOT_AUTOGEN=1 to auto-generate."
            )
            state.metadata = state.metadata or {}
            state.metadata["execution_error"] = "setup_error"
            state.metadata["error_message"] = msg
            return state
        try:
            tool_source = copilot_tool_source()
            react_solver = react(
                prompt=AgentPrompt(
                    instructions=LIVEMPCBENCH_SYSTEM_MESSAGE,
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
