from typing import Callable
from inspect_ai.scorer import (
    accuracy,
    scorer,
    stderr,
)

from openbench.metrics.simpleqa_verified import simpleqa_verified_metrics
from openbench.scorers.simpleqa import _create_simpleqa_score_fn


@scorer(metrics=[accuracy(), stderr(), simpleqa_verified_metrics()])
def simpleqa_verified_scorer(model: str) -> Callable:
    """SimpleQA Verified scorer using model grading.

    Reuses the core scoring logic from simpleqa, but with simpleqa_verified_metrics.
    """
    return _create_simpleqa_score_fn(model)
