from typing import Callable
from inspect_ai.scorer import (
    accuracy,
    choice,
    scorer,
    std,
    stderr,
)
from openbench.metrics.grouped import grouped


@scorer(metrics=[grouped(group_key="subset", metric=[accuracy(), stderr(), std()])])
def musr_grouped_scorer() -> Callable:
    """Scorer for MuSR that groups results by subset (murder_mysteries, object_placements, team_allocation)."""
    return choice()
