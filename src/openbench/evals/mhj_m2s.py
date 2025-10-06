from inspect_ai import Task, task
from openbench.datasets.mhj_m2s import get_mhj_m2s_dataset
from typing import Optional
from inspect_ai.solver import generate
from openbench.scorers.mhj_m2s import mhj_m2s_scorer


@task
def mhj_m2s(subset: Optional[str] = None) -> Task:
    """
    MHJ-M2S: Single turn conversion of the MHJ dataset

    args:
        subset: Optional[str] = None,
            The subset of the MHJ dataset to use.
            One of: "pythonize", "numberize", "hyphenize".
            If None, all subsets are used.

    Returns:
        Task: Configured MHJ-M2S task for evaluation
    """
    return Task(
        dataset=get_mhj_m2s_dataset(subset=subset),
        solver=generate(),
        scorer=mhj_m2s_scorer(),
        name="mhj_m2s",
    )


@task
def mhj_m2s_pythonize() -> Task:
    """
    MHJ-M2S: Single turn conversion (pythonize) of the MHJ dataset
    """
    return mhj_m2s(subset="pythonize")


@task
def mhj_m2s_numberize() -> Task:
    """
    MHJ-M2S: Single turn conversion (numberize) of the MHJ dataset
    """
    return mhj_m2s(subset="numberize")


@task
def mhj_m2s_hyphenize() -> Task:
    """
    MHJ-M2S: Single turn conversion (hyphenize) of the MHJ dataset
    """
    return mhj_m2s(subset="hyphenize")
