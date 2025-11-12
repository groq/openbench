"""Metrics for DocVQA evaluation.

Provides ANLS (Average Normalized Levenshtein Similarity) metrics and breakdowns
for evaluating document visual question answering performance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import cast

from inspect_ai.scorer import Metric, Value, metric
from inspect_ai.scorer._metric import SampleScore


@metric
def anls_by_question_type() -> Metric:
    """Compute ANLS score breakdown by question type.

    Different question types (figure/diagram, table, form, etc.) may have
    different difficulty levels. This metric shows performance per category.

    Does NOT include an overall score - that's provided by accuracy() metric.

    Returns:
        Metric function that computes ANLS by question type from SampleScore objects
    """

    def compute(scores: list[SampleScore]) -> Value:
        """Compute ANLS breakdown by question type.

        Args:
            scores: List of SampleScore objects from evaluation

        Returns:
            Dictionary mapping question types to their average ANLS scores
        """
        if not scores:
            return {}

        # Collect ANLS scores by question type
        type_scores: dict[str, list[float]] = defaultdict(list)

        for sample_score in scores:
            anls = cast(float, sample_score.score.value)
            metadata = sample_score.score.metadata or {}
            question_types = metadata.get("question_types", [])

            # A question can have multiple types
            if question_types:
                for qtype in question_types:
                    # Normalize to lowercase for consistent display
                    normalized_type = qtype.lower()
                    type_scores[normalized_type].append(anls)
            else:
                # Questions without specific types go to "other"
                type_scores["other"].append(anls)

        # Compute average for each type (NO overall - that's in accuracy())
        type_averages = {}
        for qtype in sorted(type_scores.keys()):
            scores_list = type_scores[qtype]
            if scores_list:
                type_averages[qtype] = sum(scores_list) / len(scores_list)

        return type_averages

    return compute
