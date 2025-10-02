"""
GPQA MC: Graduate-Level Science Questions (Multiple Choice)

This benchmark evaluates models on graduate-level science questions across physics,
chemistry, and biology. The questions are designed to be challenging even for experts
and require deep domain knowledge and reasoning.

Dataset: Idavidrein/gpqa, subset gpqa_main
Split: train (as test split is not public)

Reference: https://arxiv.org/abs/2311.12022

Note: This dataset is gated and requires authentication to access.

Sample usage:
```bash
bench eval gpqa_mc --model groq/llama-3.1-70b-versatile
```
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a GPQA record to an OpenBench MCQSample.

    The dataset contains:
    - Question: The question text
    - Correct Answer: The correct answer text
    - Incorrect Answer 1/2/3: Three incorrect answer options
    - subdomain: The scientific subdomain (e.g., Physics, Chemistry, Biology)
    """
    question = record.get("Question", record.get("question", ""))

    # Get all answer options
    correct_answer = record.get("Correct Answer", "")
    incorrect_1 = record.get("Incorrect Answer 1", "")
    incorrect_2 = record.get("Incorrect Answer 2", "")
    incorrect_3 = record.get("Incorrect Answer 3", "")

    # Place correct answer in position A (we'll shuffle if needed)
    # For now, keep it simple and fixed
    options = [correct_answer, incorrect_1, incorrect_2, incorrect_3]

    # Filter out empty options
    options = [opt for opt in options if opt]

    # Ensure we have 4 options, pad if necessary
    while len(options) < 4:
        options.append("No answer provided")

    return MCQSample(
        input=MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            prompt=question,
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            option_d=options[3],
        ),
        target="A",  # Correct answer is always in position A with this setup
        metadata={
            "subdomain": record.get("subdomain", record.get("Subdomain", "")),
        },
    )


@task
def gpqa_mc(split: str = "train") -> Task:
    """Evaluate the GPQA MC benchmark for graduate-level science questions.

    Note: Requires authentication to access the dataset.
    Set HF_TOKEN before running.
    """
    return MCQEval(
        name="gpqa_mc",
        dataset_path="Idavidrein/gpqa",
        subset_name="gpqa_main",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        group_keys=["subdomain"],  # Group metrics by scientific subdomain
    )
