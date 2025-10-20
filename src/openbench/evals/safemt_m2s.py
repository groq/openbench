"""SafeMT single-turn (m2s) jailbreak evaluation."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.solver import generate

from openbench.datasets.safemt_m2s import get_safemt_m2s_dataset
from openbench.scorers.strong_reject import strong_reject_scorer


@task
def safemt_m2s() -> Task:
    """Run the SafeMT m2s eval (optionally targeting a specific subset)."""

    return Task(
        dataset=get_safemt_m2s_dataset(),
        solver=generate(),
        scorer=strong_reject_scorer(),
        name="safemt_m2s",
    )
