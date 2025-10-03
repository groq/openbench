from openbench.datasets.mockaime import get_otis_mock_aime_dataset
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
