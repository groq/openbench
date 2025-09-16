"""Clockbench evaluation task."""

from inspect_ai import Task, task
from openbench.solvers.clockbench import clockbench_solver
from openbench.datasets.clockbench import get_clockbench_dataset
from openbench.scorers.clockbench import clockbench_scorer


@task
def clockbench() -> Task:
    """Clockbench evaluation task."""
    dataset = get_clockbench_dataset()

    return Task(
        dataset=dataset,
        solver=clockbench_solver(),
        scorer=clockbench_scorer(),
    )
