"""ChartQAPro metrics for per-category performance breakdown."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import cast

from inspect_ai.scorer import Metric, Value, metric
from inspect_ai.scorer._metric import SampleScore


@metric
def chartqapro_metrics() -> Metric:
    """
    Compute detailed ChartQAPro metrics with per-category breakdown.

    Breaks down performance by:
    - Question type (5 categories): Factoid, Multi Choice, Hypothetical, Fact Checking, Conversational
    - Overall accuracy

    Matches the official evaluation output format.

    Returns:
        Metric calculator function
    """

    def metric_calculator(scores: list[SampleScore]) -> Value:
        results = OrderedDict()

        # Aggregate by question type
        by_type = defaultdict(list)

        for sample_score in scores:
            metadata = sample_score.score.metadata or {}
            q_type = metadata.get("question_type", "unknown")
            score_val = cast(float, sample_score.score.value)

            by_type[q_type].append(score_val)

        # Overall metrics
        all_scores = [cast(float, s.score.value) for s in scores]
        results["overall"] = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Per question type (sorted alphabetically for consistency)
        for q_type in sorted(by_type.keys()):
            scores_list = by_type[q_type]
            avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
            results[q_type] = avg_score

        return results

    return metric_calculator
