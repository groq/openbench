"""
MathVista evaluation implementation.

The model must be able to handle images as input.

Sample usage:

```bash
uv run bench eval mathvista --model "groq/meta-llama/llama-4-maverick-17b-128e-instruct"
```

Dataset: https://huggingface.co/datasets/AI4Math/MathVista
Default split: testmini
"""

from inspect_ai.dataset import hf_dataset
from inspect_ai.solver import solver, TaskState, Generate, generate
from inspect_ai import Task, task
from inspect_ai.model import (
    get_model,
)
from openbench.datasets.mathvista import record_to_sample
from openbench.scorers.mathvista import custom_scorer


@solver
def custom_solver():
    model = get_model()

    async def solve(state: TaskState, generate: Generate):
        resp = await model.generate(
            input=state.input,
        )

        state.messages.append(resp.choices[0].message)
        return state

    return solve


@task
def mathvista():
    return Task(
        dataset=hf_dataset(
            "AI4Math/MathVista", split="testmini", sample_fields=record_to_sample
        ),
        solver=[generate()],
        scorer=custom_scorer(),
    )
