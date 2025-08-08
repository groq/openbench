"""HealthBench evaluation implementation."""

from typing import Optional

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.healthbench import get_dataset
from openbench.scorers.healthbench import healthbench_scorer


@task
def healthbench(
    subset: Optional[str] = None,
    grader_model: str = "openai/gpt-4.1-2025-04-14",
) -> Task:
    """HealthBench: Medical dialogue evaluation using physician-created rubrics.

    Based on the HealthBench benchmark from OpenAI's simple-evals.
    Evaluates medical dialogue completions against detailed rubrics.

    Args:
        subset: Which subset to evaluate ("hard", "consensus", or None for main)
        grader_model: Model to use for grading rubrics

    Returns:
        Task configured for HealthBench evaluation
    """
    return Task(
        dataset=get_dataset(subset=subset),
        solver=[generate()],
        scorer=healthbench_scorer(grader_model=grader_model),
        name="healthbench",
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation for medical advice
            max_tokens=8192,  # Allow longer responses for detailed medical explanations
        ),
    )


@task
def healthbench_hard(grader_model: str = "openai/gpt-4.1-2025-04-14") -> Task:
    """HealthBench Hard subset: Most challenging medical dialogue cases.

    Args:
        grader_model: Model to use for grading rubrics

    Returns:
        Task configured for HealthBench Hard evaluation
    """
    return Task(
        dataset=get_dataset(subset="hard"),
        solver=[generate()],
        scorer=healthbench_scorer(grader_model=grader_model),
        name="healthbench_hard",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=8192,
        ),
    )


@task
def healthbench_consensus(grader_model: str = "openai/gpt-4.1-2025-04-14") -> Task:
    """HealthBench Consensus subset: Cases with physician consensus.

    Args:
        grader_model: Model to use for grading rubrics

    Returns:
        Task configured for HealthBench Consensus evaluation
    """
    return Task(
        dataset=get_dataset(subset="consensus"),
        solver=[generate()],
        scorer=healthbench_scorer(grader_model=grader_model),
        name="healthbench_consensus",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=8192,
        ),
    )
