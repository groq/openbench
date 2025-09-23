"""MMStar evaluation task.

Runs multimodal model evaluation with additional text-only ablations to
compute the Multi-modal Gain (MG) and Multi-modal Leakage (ML) metrics
proposed in the MMStar benchmark.
"""

from __future__ import annotations

from typing import Optional

from inspect_ai import Task, task

from openbench.datasets.mmstar import get_mmstar_dataset
from openbench.scorers.mmstar import mmstar_scorer
from openbench.solvers.mmstar import mmstar_solver


@task
def mmstar(base_model: Optional[str] = None) -> Task:
    """Create the MMStar evaluation task.

    Args:
        base_model: Optional provider/model identifier for the text-only base LLM.
            When supplied, this model is used to compute St for the ML metric.
    """

    dataset = get_mmstar_dataset()
    return Task(
        dataset=dataset,
        solver=mmstar_solver(base_model=base_model),
        scorer=mmstar_scorer(),
        name="mmstar",
    )
