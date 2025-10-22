from inspect_ai import Task, task

from inspect_ai.solver import generate

from openbench.datasets.mhj_m2s import get_mhj_m2s_dataset
from openbench.scorers.strong_reject import strong_reject_scorer


@task
def mhj_m2s() -> Task:
    """
    MHJ-M2S: Single turn conversion of the MHJ dataset

    Returns:
        Task: Configured MHJ-M2S task for evaluation
    """
    return Task(
        dataset=get_mhj_m2s_dataset(),
        solver=generate(),
        scorer=strong_reject_scorer(),
        name="mhj_m2s",
    )
