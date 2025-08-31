"""
GitHub Multiple Choice Questions
Authored by:
Rootly AI Labs
Based on: https://huggingface.co/datasets/TheFloatingString/gmcq

# run code generation
bench eval gmcq --model "groq/llama-3.1-8b-instant" --T subtask=mastodon

If subtask is None, then the entire dataset is used.

Please refer to https://huggingface.co/datasets/TheFloatingString/gmcq for the subtask to use.
There are 6 subtasks as of Tuesday, August 19, 2025, and the None option for the entire dataset:

- bluesky
- chroma
- cloudflare
- duckdb
- mastodon
- tailscale
- None
"""

from inspect_ai import Task, task
from inspect_ai.solver import TaskState, Generate, solver
from inspect_ai.model import get_model, GenerateConfig

from openbench.scorers.rootly_gmcq import custom_scorer
from openbench.datasets.rootly_gmcq import load_dataset


@solver
def custom_solver():
    model = get_model()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        resp = await model.generate(input=state.input)
        state.messages.append(resp.choices[0].message)
        return state

    return solve


@task
def rootly_gmcq(subtask: str = None) -> Task:  # type: ignore
    dataset = load_dataset(subtask)
    return Task(
        dataset=dataset,
        solver=custom_solver(),
        scorer=custom_scorer(),
        config=GenerateConfig(),
    )
