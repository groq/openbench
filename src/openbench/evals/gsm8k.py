"""GSM8K (Grade School Math 8K) benchmark evaluation.

Dataset: openai/gsm8k
Paper: Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021)
https://arxiv.org/abs/2110.14168
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.scorers.grade_school_math import grade_school_math_scorer

PROMPT_TEMPLATE = """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{question}"""


def record_to_sample(record: dict) -> Sample:
    """Convert a GSM8K record to an Inspect Sample."""
    target = record["answer"].split("####")[-1].strip()
    return Sample(
        input=PROMPT_TEMPLATE.format(question=record["question"]),
        target=target,
        metadata={"answer_prefix": "Answer"},
    )


@task
def gsm8k(split: str = "test") -> Task:
    """GSM8K: Grade School Math 8K benchmark."""
    return Task(
        dataset=hf_dataset(
            path="gsm8k",
            data_dir="main",
            split=split,
            sample_fields=record_to_sample,
        ),
        solver=[generate()],
        scorer=grade_school_math_scorer(),
        config=GenerateConfig(temperature=0.0, max_tokens=2048),
    )
