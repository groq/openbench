from typing import Optional
from inspect_ai import task, Task
from openbench.datasets.repeated_words import get_dataset
from openbench.scorers.repeated_words import repeated_words_scorer
from openbench.solvers.dynamic_tokens import generate_with_dynamic_tokens


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
