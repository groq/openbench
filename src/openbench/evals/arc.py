"""
ARC: AI2 Reasoning Challenge

ARC is a multiple-choice question-answering dataset that tests scientific reasoning
capabilities through grade-school science exam questions. Questions are sourced from
standardized tests and have been partitioned into a Challenge Set and an Easy Set.

Dataset: allenai/ai2_arc
Paper: https://arxiv.org/abs/1803.05457

Sample usage:
```bash
bench eval arc_easy --model "groq/llama-3.1-70b"
bench eval arc_challenge --model "groq/llama-3.1-70b"
```
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert an ARC record to an OpenBench MCQSample.

    ARC records have:
    - question: The question text
    - choices: Dict with "text" (list of choices) and "label" (list of letters)
    - answerKey: The correct answer letter
    """
    question = record["question"]
    choices = record["choices"]
    answer_key = record["answerKey"]

    # ARC can have 3-5 choices
    choice_texts = choices["text"]
    choice_labels = choices["label"]

    # Create mapping from label to text
    label_to_text = dict(zip(choice_labels, choice_texts))

    # Normalize numeric labels to letters if present
    label_mapping = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    # Build options list dynamically based on available choices
    options = []
    expected_labels = ["A", "B", "C", "D", "E"]
    for i, label in enumerate(expected_labels[: len(choice_texts)]):
        # Try letter first, then numeric equivalent
        text = label_to_text.get(label) or label_to_text.get(str(i + 1), "")
        if text:
            options.append(text)

    # Build prompt with dynamic number of choices
    prompt = create_dynamic_multiple_choice_prompt(question, options)

    # Normalize answer key (sometimes it's "1" instead of "A")
    target = label_mapping.get(answer_key, answer_key)

    return MCQSample(
        input=prompt,
        target=target,
        metadata={"question": question[:100]},  # Truncate for logging
    )


@task
def arc_easy(split: str = "test") -> Task:
    """
    ARC-Easy: The easier partition of the AI2 Reasoning Challenge.

    Contains 2,376 test questions that are more straightforward.

    Args:
        split: Dataset split (default: "test")
            - "test": 2,376 questions
            - "train": 2,251 questions
            - "validation": 570 questions

    Returns:
        Task configured for ARC-Easy evaluation
    """
    return MCQEval(
        name="arc_easy",
        dataset_path="allenai/ai2_arc",
        subset_name="ARC-Easy",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        config=GenerateConfig(
            temperature=0.0,  # Deterministic for consistency
        ),
    )


@task
def arc_challenge(split: str = "test") -> Task:
    """
    ARC-Challenge: The more challenging partition of the AI2 Reasoning Challenge.

    Contains 1,172 test questions that are more difficult and require deeper reasoning.

    Args:
        split: Dataset split (default: "test")
            - "test": 1,172 questions
            - "train": 1,119 questions
            - "validation": 299 questions

    Returns:
        Task configured for ARC-Challenge evaluation
    """
    return MCQEval(
        name="arc_challenge",
        dataset_path="allenai/ai2_arc",
        subset_name="ARC-Challenge",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        config=GenerateConfig(
            temperature=0.0,  # Deterministic for consistency
        ),
    )
