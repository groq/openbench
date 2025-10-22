"""CoSafe single-turn (m2s) jailbreak evaluation."""

from __future__ import annotations


from inspect_ai import Task, task
from inspect_ai.solver import generate

from openbench.datasets.cosafe_m2s import get_cosafe_m2s_dataset
from openbench.scorers.strong_reject import strong_reject_scorer


@task
def cosafe_m2s() -> Task:
    """Run the CoSafe m2s eval"""

    return Task(
        dataset=get_cosafe_m2s_dataset(),
        solver=generate(),
        scorer=strong_reject_scorer(),
        name="cosafe_m2s",
    )
