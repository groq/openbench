"""Instruction Following evaluation implementation."""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.instruction_following import get_dataset
from openbench.scorers.instruction_following import instruction_following_scorer


@task
def instruction_following_strict() -> Task:
    """Strict instruction following evaluation.

    Tests ability to follow specific formatting and content constraints.
    Based on IFEval benchmark from Zhou et al. (2023).
    """
    return Task(
        dataset=get_dataset(),
        solver=[generate()],
        scorer=instruction_following_scorer(mode="strict"),
        name="instruction_following_strict",
        config=GenerateConfig(
            temperature=0.0,  # Deterministic
            max_tokens=2048,
        ),
    )


@task
def instruction_following_loose() -> Task:
    """Loose instruction following evaluation (more forgiving).

    Tests ability to follow constraints with some flexibility for formatting.
    """
    return Task(
        dataset=get_dataset(),
        solver=[generate()],
        scorer=instruction_following_scorer(mode="loose"),
        name="instruction_following_loose",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=2048,
        ),
    )
