"""
BoolQ: A Question Answering Dataset for Boolean Reasoning
https://arxiv.org/abs/1905.10044

Sample usage:
```bash
uv run openbench eval boolq --model "groq/llama-3.1-8b-instant"
```
The prompt is based on the default prompt (default.yaml) from
the EleutherAI/lm-evaluation-harness implementation:

https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/super_glue/boolq/default.yaml
"""

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from openbench.datasets.boolq import record_to_sample


@task
def boolq(split="validation"):
    return Task(
        dataset=hf_dataset("boolq", split=split, sample_fields=record_to_sample),
        solver=multiple_choice(),
        scorer=choice(),
    )
