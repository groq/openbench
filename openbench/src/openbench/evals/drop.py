"""DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs.

Based on the paper by Dua et al. (2019).
https://arxiv.org/abs/1903.00161
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.drop import get_dataset
from openbench.scorers.drop import drop_scorer


@task
def drop(
    num_examples: int | None = None,
    train_samples_per_prompt: int = 3,
) -> Task:
    """DROP: Reading comprehension requiring discrete reasoning.

    A reading comprehension benchmark that requires discrete reasoning over
    paragraphs, including arithmetic, counting, and sorting operations.

    Args:
        num_examples: Number of examples to evaluate (None for all)
        train_samples_per_prompt: Number of few-shot examples (default: 3)

    Returns:
        Task configured for DROP evaluation
    """
    return Task(
        dataset=get_dataset(
            num_examples=num_examples,
            train_samples_per_prompt=train_samples_per_prompt,
        ),
        solver=[generate()],
        scorer=drop_scorer(),
        name="drop",
        config=GenerateConfig(
            temperature=0.0,  # Deterministic for reasoning tasks
            max_tokens=8192,  # Allow for reasoning steps
        ),
    )
