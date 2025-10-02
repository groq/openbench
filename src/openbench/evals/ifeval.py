"""Instruction Following evaluation implementation."""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.ifeval import get_dataset
from openbench.scorers.ifeval import ifeval_combined_scorer


@task
def ifeval() -> Task:
    """Combined instruction following evaluation.

    Tests ability to follow specific formatting and content constraints.
    Shows both strict and loose evaluation metrics.
    Based on IFEval benchmark from Zhou et al. (2023).
    """
    return Task(
        dataset=get_dataset(),
        solver=[generate()],
        scorer=ifeval_combined_scorer(),
        name="ifeval",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=2048,
        ),
    )
