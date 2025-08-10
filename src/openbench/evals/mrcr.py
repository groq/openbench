from typing import Optional
from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from openbench.datasets.mrcr import get_dataset
from openbench.scorers.mrcr import mrcr_scorer


@task
def openai_mrcr(max_context_size: Optional[int] = None) -> Task:
    """Memory-Recall with Contextual Retrieval (MRCR).

    Evaluates retrieval and recall in long contexts by placing 2, 4 or 8 needles in the prompt and measuring whether the
    model can correctly extract and use them.

    Args:
        max_context_size: Maximum context size in tokens. Defaults to None.

    Returns:
        Task configured for MRCR evaluation.
    """

    return Task(
        dataset=get_dataset(max_context_size=max_context_size),
        solver=generate(),
        scorer=mrcr_scorer(),
        name="openai_mrcr",
        config=GenerateConfig(temperature=0.0),
    )


@task
def openai_mrcr_2n(max_context_size: Optional[int] = None) -> Task:
    """Memory-Recall with Contextual Retrieval (MRCR).

    Evaluates retrieval and recall in long contexts by placing 2 needles in the prompt and measuring whether the
    model can correctly extract and use them.

    Args:
        max_context_size: Maximum context size in tokens. Defaults to None.

    Returns:
        Task configured for MRCR 2 needles evaluation.
    """

    return Task(
        dataset=get_dataset(needles=2, max_context_size=max_context_size),
        solver=generate(),
        scorer=mrcr_scorer(),
        name="openai_mrcr_2n",
        config=GenerateConfig(temperature=0.0),
    )


@task
def openai_mrcr_4n(max_context_size: Optional[int] = None) -> Task:
    """Memory-Recall with Contextual Retrieval (MRCR).

    Evaluates retrieval and recall in long contexts by placing 4 needles in the prompt and measuring whether the
    model can correctly extract and use them.

    Args:
        max_context_size: Maximum context size in tokens. Defaults to None.

    Returns:
        Task configured for MRCR 4 needles evaluation.
    """

    return Task(
        dataset=get_dataset(needles=4, max_context_size=max_context_size),
        solver=generate(),
        scorer=mrcr_scorer(),
        name="openai_mrcr_4n",
        config=GenerateConfig(temperature=0.0),
    )


@task
def openai_mrcr_8n(max_context_size: Optional[int] = None) -> Task:
    """Memory-Recall with Contextual Retrieval (MRCR).

    Evaluates retrieval and recall in long contexts by placing 8 needles in the prompt and measuring whether the
    model can correctly extract and use them.

    Args:
        max_context_size: Maximum context size in tokens. Defaults to None.

    Returns:
        Task configured for MRCR 8 needles evaluation.
    """

    return Task(
        dataset=get_dataset(needles=8, max_context_size=max_context_size),
        solver=generate(),
        scorer=mrcr_scorer(),
        name="openai_mrcr_8n",
        config=GenerateConfig(temperature=0.0),
    )
