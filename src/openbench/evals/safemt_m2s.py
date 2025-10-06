"""SafeMT single-turn (m2s) jailbreak evaluation."""

from __future__ import annotations

from typing import Optional

from inspect_ai import Task, task
from inspect_ai.solver import generate

from openbench.datasets.safemt_m2s import get_safemt_m2s_dataset
from openbench.scorers.score_reject import score_reject_scorer


@task
def safemt_m2s(subset: Optional[str] = None) -> Task:
    """Run the SafeMT m2s eval (optionally targeting a specific subset)."""

    return Task(
        dataset=get_safemt_m2s_dataset(subset=subset),
        solver=generate(),
        scorer=score_reject_scorer(),
        name="safemt_m2s",
    )


@task
def safemt_m2s_pythonize() -> Task:
    """SafeMT m2s eval restricted to the pythonize m2s method."""

    return safemt_m2s(subset="pythonize")


@task
def safemt_m2s_numberize() -> Task:
    """SafeMT m2s eval restricted to the numberize m2s method."""

    return safemt_m2s(subset="numberize")


@task
def safemt_m2s_hyphenize() -> Task:
    """SafeMT m2s eval restricted to the hyphenize m2s method."""

    return safemt_m2s(subset="hyphenize")
