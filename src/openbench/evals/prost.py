"""
PROST: Physical Reasoning about Object States

This benchmark evaluates models on physical reasoning about everyday objects and their properties.
Models must predict which object would be most/least likely to exhibit certain physical behaviors
(rolling, bouncing, breaking, etc.).

Dataset: lighteval/prost
Split: test (18,736 samples)

Reference: https://arxiv.org/abs/2106.03634

Sample usage:
```bash
bench eval prost --model groq/llama-3.1-70b-versatile
```
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a PROST record to an OpenBench MCQSample.

    The dataset contains:
    - context: Description of the physical scenario
    - question: Question with [MASK] placeholder
    - ex_question: Explicit version of the question
    - A, B, C, D: Four object options
    - label: Index (0-3) of correct answer
    - group: Physical property being tested (e.g., 'rolling', 'bouncing')
    """
    context = record["context"]
    question = record["ex_question"]  # Use explicit question instead of masked version

    # Combine context and question
    full_question = f"{context}\n\n{question}"

    # Get options
    option_a = record["A"]
    option_b = record["B"]
    option_c = record["C"]
    option_d = record["D"]

    # Convert label index to letter (0->A, 1->B, 2->C, 3->D)
    label_idx = record["label"]
    target = chr(65 + label_idx)  # 65 is ASCII for 'A'

    return MCQSample(
        input=MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            prompt=full_question,
            option_a=option_a,
            option_b=option_b,
            option_c=option_c,
            option_d=option_d,
        ),
        target=target,
        metadata={
            "group": record.get("group", ""),
        },
    )


@task
def prost(split: str = "test") -> Task:
    """Evaluate the PROST benchmark for physical reasoning about object states."""
    return MCQEval(
        name="prost",
        dataset_path="lighteval/prost",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        group_keys=["group"],  # Group metrics by physical property
    )
