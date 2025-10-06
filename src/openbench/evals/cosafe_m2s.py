"""CoSafe single-turn (m2s) jailbreak evaluation."""

from __future__ import annotations

from typing import Optional

from inspect_ai import Task, task
from inspect_ai.solver import generate

from openbench.datasets.cosafe_m2s import get_cosafe_m2s_dataset
from openbench.scorers.score_reject import score_reject_scorer


@task
def cosafe_m2s(subset: Optional[str] = None) -> Task:
    """Run the CoSafe m2s eval (optionally targeting a specific subset)."""

    return Task(
        dataset=get_cosafe_m2s_dataset(subset=subset),
        solver=generate(),
        scorer=score_reject_scorer(),
        name="cosafe_m2s",
    )


@task
def cosafe_m2s_pythonize() -> Task:
    """CoSafe m2s eval restricted to the pythonize subset."""

    return cosafe_m2s(subset="pythonize")


@task
def cosafe_m2s_numberize() -> Task:
    """CoSafe m2s eval restricted to the numberize subset."""

    return cosafe_m2s(subset="numberize")


@task
def cosafe_m2s_hyphenize() -> Task:
    """CoSafe m2s eval restricted to the hyphenize subset."""

    return cosafe_m2s(subset="hyphenize")
