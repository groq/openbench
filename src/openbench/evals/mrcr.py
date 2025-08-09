from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from openbench.datasets.mrcr import get_dataset
from openbench.scorers.mrcr import mrcr_scorer


@task
def openai_mrcr() -> Task:
    """Memory-Recall with Contextual Retrieval (MRCR).

    Evaluates retrieval and recall in long contexts by placing a specified
    number of "needles" (facts) in the prompt and measuring whether the
    model can correctly extract and use them.

    Args:
        None

    Returns:
        Task configured for MRCR evaluation.
    """

    return Task(
        dataset=get_dataset(),
        solver=generate(),
        scorer=mrcr_scorer(),
        name="openai_mrcr",
        config=GenerateConfig(temperature=0.0),
    )


@task
def openai_mrcr_2n() -> Task:
    """Memory-Recall with Contextual Retrieval (MRCR).

    Evaluates retrieval and recall in long contexts by placing a specified
    number of "needles" (facts) in the prompt and measuring whether the
    model can correctly extract and use them.

    Args:
        None

    Returns:
        Task configured for MRCR evaluation.
    """

    return Task(
        dataset=get_dataset(needles=2),
        solver=generate(),
        scorer=mrcr_scorer(),
        name="openai_mrcr_2n",
        config=GenerateConfig(temperature=0.0),
    )


@task
def openai_mrcr_4n() -> Task:
    """Memory-Recall with Contextual Retrieval (MRCR).

    Evaluates retrieval and recall in long contexts by placing a specified
    number of "needles" (facts) in the prompt and measuring whether the
    model can correctly extract and use them.

    Args:
        None

    Returns:
        Task configured for MRCR evaluation.
    """

    return Task(
        dataset=get_dataset(needles=4),
        solver=generate(),
        scorer=mrcr_scorer(),
        name="openai_mrcr_4n",
        config=GenerateConfig(temperature=0.0),
    )


@task
def openai_mrcr_8n() -> Task:
    """Memory-Recall with Contextual Retrieval (MRCR).

    Evaluates retrieval and recall in long contexts by placing a specified
    number of "needles" (facts) in the prompt and measuring whether the
    model can correctly extract and use them.

    Args:
        None

    Returns:
        Task configured for MRCR evaluation.
    """

    return Task(
        dataset=get_dataset(needles=8),
        solver=generate(),
        scorer=mrcr_scorer(),
        name="openai_mrcr_8n",
        config=GenerateConfig(temperature=0.0),
    )
