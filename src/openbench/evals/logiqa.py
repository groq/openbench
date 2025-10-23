"""
LogiQA - Logical Reasoning Dataset

LogiQA is a dataset for testing logical reasoning and critical thinking. It consists
of 8,678 QA instances drawn from publicly available questions of the National Civil
Servants Examination of China, covering multiple types of deductive reasoning.

Sample usage:
```bash
bench eval logiqa --model "groq/llama-3.1-70b"
```

Citation:
@inproceedings{liu2020logiqa,
    title={LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning},
    author={Liu, Jian and Cui, Leyang and Liu, Hanmeng and Huang, Dandan and Wang, Yile and Zhang, Yue},
    booktitle={Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence},
    year={2020}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a LogiQA record to an OpenBench MCQSample."""
    # Build the question with context
    question = f"{record['context']}\n\n{record['question']}"

    # Get the options (4-way multiple choice)
    choices = record["options"]

    # Create the multiple choice prompt
    prompt = create_dynamic_multiple_choice_prompt(question, choices)

    # Convert label to uppercase letter (a -> A, b -> B, etc.)
    target = record["label"].upper()

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "context": record["context"],
            "question": record["question"],
        },
    )


@task
def logiqa(split: str = "validation") -> Task:
    """
    Evaluate the LogiQA benchmark for logical reasoning.

    Args:
        split: Dataset split to use ("train", "validation", "test")
               Default: "validation" (standard evaluation split)

    Returns:
        Task: Inspect AI task for LogiQA evaluation
    """
    valid_splits = ["train", "validation", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    return MCQEval(
        name="logiqa",
        dataset_path="lighteval/logiqa_harness",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        dataset_kwargs={},
    )
