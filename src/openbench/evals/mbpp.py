from __future__ import annotations

from typing import Optional

from inspect_ai import Task, task, Epochs
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.mbpp import get_dataset
from openbench.scorers.mbpp import mbpp_verify


@task
def mbpp(
    limit: Optional[int] = None,
    # execution / scoring
    timeout: int = 6,  # seconds for python -c execution in sandbox
    epochs: int = 1,
    temperature: float = 0.0,  # determinism for code generation
) -> Task:
    return Task(
        dataset=get_dataset(limit=limit),
        solver=generate(),
        scorer=mbpp_verify(timeout=timeout),
        sandbox="local",
        config=GenerateConfig(temperature=temperature),
        epochs=Epochs(epochs),
        name="mbpp",
    )
