from typing import Optional
from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from openbench.datasets.multichallenge import get_dataset
from openbench.scorers.multichallenge import multichallenge_scorer


@task
def multichallenge(
    grader_model: str = "openai/gpt-4.1-2025-04-14",
    max_turns: Optional[int] = None,
) -> Task:
    """
    MultiChallenge evaluation task.

    Loads the full dataset (optionally truncated), sends conversations to the model,
    and computes strict per-axis metrics using a judging model.

    Args:
        grader_model: Model to use for grading responses (defaults to gpt-4.1-2025-04-14)
        max_turns: Truncate conversations to last N turns.

    Returns:
        Configured Task object.
    """

    return Task(
        dataset=get_dataset(max_turns=max_turns),
        solver=[generate()],
        scorer=multichallenge_scorer(model=grader_model),
        name="multichallenge",
        config=GenerateConfig(temperature=0.0),
    )
