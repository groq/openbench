"""MMMU mixed-type scorer that dispatches based on question_type.

Uses MCQ scoring for multiple-choice items and open-answer scorer for open items.
Supports grouped metrics across both types.
"""

from typing import Callable, Optional, List, Any

from inspect_ai.scorer import scorer, accuracy, stderr, std, Score, Target
from inspect_ai.solver import TaskState

from openbench.scorers.mcq import create_mcq_scorer
from openbench.scorers.open_answer import create_open_answer_scorer
from openbench.metrics.grouped import grouped


def mmmu_mixed_scorer(
    group_keys: Optional[List[str]] = None,
) -> Callable:
    """Create a scorer that handles both MCQ and open-answer MMMU items.

    Args:
        group_keys: Optional metadata keys to group metrics by.
    """

    mcq_scorer = create_mcq_scorer(group_keys=group_keys)()
    open_scorer = create_open_answer_scorer(group_keys=group_keys)()

    metrics: List[Any] = [accuracy(), stderr(), std()]

    if group_keys:
        for key in group_keys:
            metrics.append(grouped(group_key=key, metric=[accuracy(), stderr(), std()]))

    @scorer(metrics=metrics)
    def mixed() -> Callable:
        async def score(state: TaskState, target: Target) -> Score:
            qtype = str(state.metadata.get("question_type", "multiple-choice")).lower()
            if qtype == "open":
                return await open_scorer(state, target)
            return await mcq_scorer(state, target)

        return score

    return mixed
