"""
ChartQA evaluation implementation.

The model must be able to handle images as input.

Dataset: https://huggingface.co/datasets/HuggingFaceM4/ChartQA
Default split: test

Sample usage:

```bash
uv run bench eval chartqa --model "groq/meta-llama/llama-4-maverick-17b-128e-instruct"
```
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate
from openbench.datasets.chartqa import load_dataset
from openbench.scorers.chartqa import custom_scorer


@task
def chartqa(split: str = "test"):
    return Task(
        dataset=load_dataset(split),
        solver=[generate()],
        scorer=custom_scorer(),
    )
