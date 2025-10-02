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
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE


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

    # ARC typically has 3-5 choices, map them to A, B, C, D, E
    choice_texts = choices["text"]
    choice_labels = choices["label"]

    # Create mapping from label to text
    label_to_text = dict(zip(choice_labels, choice_texts))

    # Pad with empty strings for missing options (up to 5 choices)
    option_a = label_to_text.get("A", label_to_text.get("1", ""))
    option_b = label_to_text.get("B", label_to_text.get("2", ""))
    option_c = label_to_text.get("C", label_to_text.get("3", ""))
    option_d = label_to_text.get("D", label_to_text.get("4", ""))

    # Build prompt with only non-empty options
    prompt = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        prompt=question,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )

    # Normalize answer key (sometimes it's "1" instead of "A")
    answer_mapping = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    target = answer_mapping.get(answer_key, answer_key)

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
