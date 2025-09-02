from typing import Optional
from inspect_ai import task, Task
from inspect_ai.solver import Solver, solver
from inspect_ai.model import GenerateConfig
from openbench.datasets.repeated_words import get_dataset
from openbench.scorers.repeated_words import repeated_words_scorer


@solver
def generate_with_dynamic_tokens() -> Solver:
    """Custom solver that sets max_tokens per sample based on metadata."""

    async def solve(state, generate_fn):
        sample_max_tokens = state.metadata.get("max_output_tokens")

        config = GenerateConfig(temperature=0.0, max_tokens=sample_max_tokens)

        return await generate_fn(state, config=config)

    return solve


@task
def repeated_words(max_context_size: Optional[int] = None) -> Task:
    """Repeated Words

    One of the experiments from Chroma's Context Rot paper.

    Args:
        max_context_size: Maximum context size in tokens. Defaults to None.

    Returns:
        Task configured for Repeated Words evaluation
    """
    return Task(
        dataset=get_dataset(max_context_size=max_context_size),
        solver=[generate_with_dynamic_tokens()],
        scorer=repeated_words_scorer(),
        name="repeated_words",
    )
