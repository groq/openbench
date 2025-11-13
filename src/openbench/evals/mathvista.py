"""MathVista: Mathematical Reasoning in Visual Contexts.

Paper: https://arxiv.org/abs/2310.02255
Dataset: https://huggingface.co/datasets/AI4Math/MathVista
GitHub: https://github.com/lupantech/MathVista

MathVista is a benchmark for evaluating mathematical reasoning in visual contexts.
It contains 6,141 examples across diverse visual contexts requiring fine-grained
visual understanding and mathematical reasoning.
"""

from typing import Optional

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.mathvista import get_dataset
from openbench.scorers.mathvista import mathvista_scorer


@task
def mathvista(
    split: str = "testmini",
    question_type: Optional[str] = None,
    shuffle: bool = True,
    seed: int = 42,
    grader_model: str = "openai/gpt-4-turbo",
) -> Task:
    """MathVista: Mathematical Reasoning in Visual Contexts.

    Args:
        split: Dataset split ("testmini" or "test")
            - testmini: ~1,000 samples for development/quick testing
            - test: ~6,000 samples for full evaluation
        question_type: Optional filter ("multi_choice" or "free_form")
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        grader_model: Model to use for LLM-based answer extraction fallback
                     (default: gpt-4-turbo, matching original paper)

    Returns:
        Task for evaluation
    """
    dataset = get_dataset(
        split=split,
        question_type=question_type,
        shuffle=shuffle,
        seed=seed,
    )

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=mathvista_scorer(grader_model=grader_model)(),
        name=f"mathvista_{split}",
        config=GenerateConfig(
            max_tokens=2048,  # Allow longer responses for reasoning
        ),
    )
