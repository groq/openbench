from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from openbench.datasets.mrcr import get_dataset
from openbench.scorers.mrcr import mrcr_scorer


@task
def mrcr(needles: int = 2) -> Task:
    """Memory-Recall with Contextual Retrieval (MRCR).

    Evaluates retrieval and recall in long contexts by placing a specified
    number of "needles" (facts) in the prompt and measuring whether the
    model can correctly extract and use them.

    Args:
        needles: Number of needles in the context (allowed: 2, 4, 8). Defaults to 2.

    Returns:
        Task configured for MRCR evaluation.
    """

    if needles not in {2, 4, 8}:
        raise ValueError("'needles' must be one of {2, 4, 8}")

    return Task(
        dataset=get_dataset(needles=needles),
        solver=generate(),
        scorer=mrcr_scorer(),
        name="mrcr",
        config=GenerateConfig(temperature=0.0),
    )
