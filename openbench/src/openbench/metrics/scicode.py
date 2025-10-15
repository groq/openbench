"""SciCode metrics for evaluation scoring."""

from typing import List
from inspect_ai.scorer import Metric, Score, metric


@metric
def sub_problem_correctness() -> Metric:
    def metric(scores: List[Score]) -> int | float:
        total_correct = 0
        total_steps = 0
        for score in scores:
            total_correct += score.value["Total Correct"]  # type: ignore
            total_steps += score.value["Total Steps"]  # type: ignore
        return total_correct / total_steps

    return metric
