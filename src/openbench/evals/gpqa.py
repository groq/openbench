"""
GPQA: Graduate-Level Science Questions (Multiple Choice)

This benchmark evaluates models on graduate-level science questions across physics,
chemistry, and biology. The questions are designed to be challenging even for experts
and require deep domain knowledge and reasoning.

Dataset: Idavidrein/gpqa, subset gpqa_main
Split: train (as test split is not public)

Reference: https://arxiv.org/abs/2311.12022

Note: This dataset is gated and requires authentication to access.

Sample usage:
```bash
bench eval gpqa --model groq/llama-3.1-70b-versatile
```
"""

import random
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

    # Validate that correct answer is not empty
    if not correct_answer or not correct_answer.strip():
        raise ValueError("Correct answer is empty or missing")

    # Create list of (answer_text, is_correct) tuples
    all_answers = [
        (correct_answer, True),
        (incorrect_1, False),
        (incorrect_2, False),
        (incorrect_3, False),
    ]

    # Filter out empty answers
    all_answers = [
        (text, is_correct) for text, is_correct in all_answers if text and text.strip()
    ]

    # Pad to 4 answers BEFORE shuffling so placeholders are randomized too
    while len(all_answers) < 4:
        all_answers.append(("No answer provided", False))

    # Shuffle answers deterministically based on question text
    # This ensures consistent ordering across runs while varying across samples
    rng = random.Random(hash(question) % (2**32))
    rng.shuffle(all_answers)

    # Find position of correct answer
    correct_idx = next(i for i, (_, is_correct) in enumerate(all_answers) if is_correct)
    target = chr(65 + correct_idx)  # 0->A, 1->B, 2->C, 3->D

    # Extract answer texts (already length 4)
    options = [text for text, _ in all_answers]

    return MCQSample(
        input=MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            prompt=question,
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            option_d=options[3],
        ),
        target=target,
        metadata={
            "subdomain": record.get("subdomain", record.get("Subdomain", "")),
        },
    )


@task
def gpqa(split: str = "train") -> Task:
    """Evaluate the GPQA benchmark for graduate-level science questions.

    Note: Requires authentication to access the dataset.
    Set HF_TOKEN before running.
    """
    return MCQEval(
        name="gpqa",
        dataset_path="Idavidrein/gpqa",
        subset_name="gpqa_main",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        group_keys=["subdomain"],  # Group metrics by scientific subdomain
    )
