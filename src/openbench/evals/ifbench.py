"""IFBench evaluation."""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from openbench.datasets.ifbench import get_dataset
from openbench.scorers.ifbench import ifbench_scorer


@task
def ifbench() -> Task:
    """IFBench: challenging instruction-following constraints (strict + loose)."""
    return Task(
        dataset=get_dataset(),
        solver=[generate()],
        scorer=ifbench_scorer(),
        name="ifbench",
    )
