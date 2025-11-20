from pathlib import Path
import json

from inspect_ai import task, Task
from inspect_ai.solver import solver, Solver
from inspect_ai.agent import react, AgentPrompt
from inspect_ai.solver._task_state import TaskState
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import MemoryDataset
from inspect_ai.scorer import match

from openbench.tools.progressive_mcp_bench.mock_toolsource import mock_mcp_tool_source
from openbench.tools.progressive_mcp_bench.prefill import get_samples, DB_PATH, MCP_ROOT
from openbench.utils.text import LIVEMCPBENCH_SYSTEM_MESSAGE, PROGRESSIVE_MCP_ALL_SYSTEM_MESSAGE

from inspect_ai.scorer import scorer, accuracy, stderr, Target, Score
import re

@scorer(metrics=[accuracy(), stderr()])
def xml_tag_match(tag: str = "answer"):
    async def score(state: TaskState, target: Target):
        completion = state.output.completion
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, completion, re.DOTALL)
        
        targets = target.target
        if isinstance(targets, str):
            targets = [targets]

        if match:
            extracted = match.group(1).strip()
            # Exact match check
            is_correct = any(t.strip() == extracted for t in targets)
            return Score(value=1.0 if is_correct else 0.0, answer=extracted)
        else:
            return Score(value=0.0, explanation=f"Tag <{tag}> not found in output: {completion[:100]}...")

    return score

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
        
        # Choose system message based on strategy
        system_msg = LIVEMCPBENCH_SYSTEM_MESSAGE
        if strategy == "all":
            system_msg = PROGRESSIVE_MCP_ALL_SYSTEM_MESSAGE

        agent = react(
            prompt=AgentPrompt(
                instructions=system_msg,
            ),
            tools=[tool_source],
        )
        try:
            return await agent(state)
        except Exception as e:
            # If the agent crashes (e.g. due to APIError from malformed tool calls),
            # we catch it and return a failed state so the eval continues.
            state.output.completion = f"FAILED: Agent crashed with error: {str(e)}"
            return state

    return solve

@task
def progressive_mcp_bench_all(
    grader_model: str = "openai/gpt-4o",
) -> Task:
    return Task(
        dataset=get_dataset_with_expectations(),
        solver=[progressive_mock_solver(strategy="all")],
        scorer=xml_tag_match(),
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
