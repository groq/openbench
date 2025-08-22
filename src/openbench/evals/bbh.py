"""
Big Bench Hard implementation.

This implementation is based on the following repository,
and please refer to this repository's JSONL files for the subtasks.
https://github.com/suzgunmirac/BIG-Bench-Hard

Example usage using chain of thought (CoT) prompts:

```bash
bench eval bbh --model "groq/llama-3.1-8b-instant" -T subtask=sports_understanding
```

Example usage using direct prompts:

```bash
bench eval bbh --model "groq/llama-3.1-8b-instant" -T subtask=sports_understanding,cot=False
```

"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import TaskState, Generate, solver
from typing import Union


from openbench.scorers.bbh import custom_scorer
from openbench.datasets.bbh import load_dataset


@solver
def custom_solver():
    model = get_model()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        resp = await model.generate(input=state.input)
        state.messages.append(resp.choices[0].message)
        return state

    return solve


@task
def bbh(subtask: Union[str, None] = None, cot: str = "True") -> Task:
    """
    Big Bench Hard implementation.
    """

    return Task(
        dataset=load_dataset(subtask, cot),  # type: ignore
        solver=custom_solver(),
        scorer=custom_scorer(),
        config=GenerateConfig(),
    )
