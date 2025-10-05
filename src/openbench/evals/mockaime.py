from openbench.datasets.mockaime import (
    get_otis_mock_aime_dataset,
    get_otis_mock_aime_2024_dataset,
    get_otis_mock_aime_2025_dataset,
)
from openbench.scorers.mockaime import otis_mock_aime_scorer
from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig


@task
def otis_mock_aime() -> Task:
    """
    MockAIME evaluation task for mathematical competition problems.

    This benchmark evaluates language models on problems from the OTIS Mock AIME
    2024-2025 exams.

    Returns:
        Task object configured for MockAIME evaluation
    """
    return Task(
        dataset=get_otis_mock_aime_dataset(),
        solver=[generate()],
        scorer=otis_mock_aime_scorer(),
        config=GenerateConfig(
            max_tokens=8192,
            temperature=0.0,
        ),
    )


@task
def otis_mock_aime_2024() -> Task:
    """
    MockAIME 2024 evaluation task for mathematical competition problems.

    This benchmark evaluates language models on problems from the OTIS Mock AIME
    2024 exam only.

    Returns:
        Task object configured for MockAIME 2024 evaluation
    """
    return Task(
        dataset=get_otis_mock_aime_2024_dataset(),
        solver=[generate()],
        scorer=otis_mock_aime_scorer(),
        config=GenerateConfig(
            max_tokens=8192,
            temperature=0.0,
        ),
    )


@task
def otis_mock_aime_2025() -> Task:
    """
    MockAIME 2025 evaluation task for mathematical competition problems.

    This benchmark evaluates language models on problems from the OTIS Mock AIME
    2025 exams only.

    Returns:
        Task object configured for MockAIME 2025 evaluation
    """
    return Task(
        dataset=get_otis_mock_aime_2025_dataset(),
        solver=[generate()],
        scorer=otis_mock_aime_scorer(),
        config=GenerateConfig(
            max_tokens=8192,
            temperature=0.0,
        ),
    )
