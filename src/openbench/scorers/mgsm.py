"""MGSM scorer for evaluating multilingual math problem solutions."""

from typing import Callable

from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from openbench.metrics.mgsm import language_accuracy
from openbench.scorers.grade_school_math import score_numeric_answer


@scorer(metrics=[accuracy(), stderr(), language_accuracy()])
def mgsm_scorer() -> Callable:
    """MGSM scorer with language-specific accuracy metrics."""

    async def score(state: TaskState, target: Target) -> Score:
        result = await score_numeric_answer(state, target)
        # Add language to metadata for language_accuracy metric
        language = state.metadata.get("language", "en")
        if result.metadata is None:
            result.metadata = {}
        result.metadata["language"] = language
        return result

    return score
