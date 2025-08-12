from inspect_ai import task, Task
from inspect_ai.solver import generate, system_message
from inspect_ai.model import GenerateConfig
from openbench.datasets.hle import get_dataset
from openbench.scorers.hle import hle_scorer


# HLE system prompt as used in the original implementation
HLE_SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


@task
def hle(
    grader_model: str = "openai/o3-mini-2025-01-31", max_tokens: int = 8192
) -> Task:
    """Humanity's Last Exam: A benchmark at the frontier of human knowledge.

    HLE consists of 2,500 questions across dozens of subjects including mathematics,
    humanities, and natural sciences. Questions are designed by subject-matter experts
    globally and include both multiple-choice and short-answer formats.

    Args:
        grader_model: Model to use for grading responses (defaults to o3-mini-2025-01-31)
        max_tokens: Maximum tokens for model response (defaults to 8192 as recommended by HLE)

    Returns:
        Task configured for HLE evaluation
    """
    return Task(
        dataset=get_dataset(text_only=False),
        solver=[
            system_message(HLE_SYSTEM_PROMPT),
            generate(),
        ],
        scorer=hle_scorer(model=grader_model),
        name="hle",
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation as per HLE
            max_tokens=max_tokens,  # HLE recommends at least 8192 for reasoning models
        ),
    )


@task
def hle_text(
    grader_model: str = "openai/o3-mini-2025-01-31", max_tokens: int = 8192
) -> Task:
    """Humanity's Last Exam (Text-Only): HLE with multi-modal questions filtered out.

    This variant includes only text-based questions from HLE, excluding any questions
    that require image understanding. Useful for evaluating models without vision capabilities.

    Args:
        grader_model: Model to use for grading responses (defaults to o3-mini-2025-01-31)
        max_tokens: Maximum tokens for model response (defaults to 8192 as recommended by HLE)

    Returns:
        Task configured for HLE text-only evaluation
    """
    return Task(
        dataset=get_dataset(text_only=True),
        solver=[
            system_message(HLE_SYSTEM_PROMPT),
            generate(),
        ],
        scorer=hle_scorer(model=grader_model),
        name="hle_text",
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation as per HLE
            max_tokens=max_tokens,  # HLE recommends at least 8192 for reasoning models
        ),
    )
