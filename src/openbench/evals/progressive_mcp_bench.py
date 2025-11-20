from pathlib import Path
import json
import os

from inspect_ai import task, Task
from inspect_ai.solver import solver, Solver
from inspect_ai.agent import react, AgentPrompt
from inspect_ai.solver._task_state import TaskState
from inspect_ai.model import GenerateConfig
from inspect_ai.tool import ToolError
from inspect_ai.scorer import includes, match
from inspect_ai.dataset import MemoryDataset

from openbench.tools.progressive_mcp_bench.mock_toolsource import mock_mcp_tool_source
from openbench.tools.progressive_mcp_bench.prefill import get_samples, DB_PATH, MCP_ROOT
from openbench.utils.text import LIVEMCPBENCH_SYSTEM_MESSAGE

EXPECTATIONS_PATH = MCP_ROOT / "expectations.json"

def get_dataset_with_expectations():
    samples = get_samples()
    if EXPECTATIONS_PATH.exists():
        with open(EXPECTATIONS_PATH) as f:
            expectations = json.load(f)
        
        for sample in samples:
            if sample.id in expectations:
                sample.target = expectations[sample.id]
    
    return MemoryDataset(samples)

@solver
def progressive_mock_solver(strategy: str = "all") -> Solver:
    async def solve(state: TaskState, generate) -> TaskState:
        relevant_server = state.metadata.get("server")
        
        tool_source = mock_mcp_tool_source(
            str(DB_PATH), 
            strategy=strategy,
            relevant_server=relevant_server
        )
        
        # We use the same system message as livemcpbench/prefill
        agent = react(
            prompt=AgentPrompt(
                instructions=LIVEMCPBENCH_SYSTEM_MESSAGE,
            ),
            tools=[tool_source],
        )
        return await agent(state)

    return solve

@task
def progressive_mcp_bench_all(
    grader_model: str = "openai/gpt-4o",
) -> Task:
    return Task(
        dataset=get_dataset_with_expectations(),
        solver=[progressive_mock_solver(strategy="all")],
        scorer=match(),
        name="progressive_mcp_bench_all",
        config=GenerateConfig(
            temperature=0.0,
        ),
    )

@task
def progressive_mcp_bench_relevant(
    grader_model: str = "openai/gpt-4o",
) -> Task:
    return Task(
        dataset=get_dataset_with_expectations(),
        solver=[progressive_mock_solver(strategy="all-relevant")],
        scorer=match(),
        name="progressive_mcp_bench_relevant",
        config=GenerateConfig(
            temperature=0.0,
        ),
    )

@task
def progressive_mcp_bench_minimal(
    grader_model: str = "openai/gpt-4o",
) -> Task:
    return Task(
        dataset=get_dataset_with_expectations(),
        solver=[progressive_mock_solver(strategy="minimal")],
        scorer=match(),
        name="progressive_mcp_bench_minimal",
        config=GenerateConfig(
            temperature=0.0,
        ),
    )
