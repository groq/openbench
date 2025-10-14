"""SimpleQA Verified metrics for evaluation scoring."""

from typing import List
from inspect_ai.scorer import (
    metric,
    Metric,
    Value,
    SampleScore,
)


@metric
def simpleqa_verified_metrics() -> Metric:
    """Calculate SimpleQA Verified specific metrics: F1 and accuracy_given_attempted."""

    def metric_calculator(scores: List[SampleScore]) -> Value:
        if not scores:
            return {
                "is_correct": 0.0,
                "is_incorrect": 0.0,
                "is_not_attempted": 0.0,
                "is_given_attempted": 0.0,
                "accuracy_given_attempted": 0.0,
                "f1": 0.0,
            }

        # Count each grade type
        grade_counts = {"correct": 0, "incorrect": 0, "not_attempted": 0}

        for sample_score in scores:
            metadata = sample_score.score.metadata
            grade = metadata.get("grade", "").lower() if metadata else ""
            if grade in grade_counts:
                grade_counts[grade] += 1

        total = len(scores)
        correct_count = grade_counts["correct"]
        incorrect_count = grade_counts["incorrect"]
        not_attempted_count = grade_counts["not_attempted"]
        attempted_count = correct_count + incorrect_count

        is_correct = correct_count / total
        is_incorrect = incorrect_count / total
        is_not_attempted = not_attempted_count / total
        is_given_attempted = attempted_count / total

        # Calculate accuracy_given_attempted
        accuracy_given_attempted = (
            correct_count / attempted_count if attempted_count > 0 else 0.0
        )

        # Calculate F1
        mean_correct = is_correct
        f1 = (
            2
            * accuracy_given_attempted
            * mean_correct
            / (accuracy_given_attempted + mean_correct)
            if (accuracy_given_attempted + mean_correct) > 0
            else 0.0
        )

        return {
            "is_correct": is_correct,
            "is_incorrect": is_incorrect,
            "is_not_attempted": is_not_attempted,
            "is_given_attempted": is_given_attempted,
            "accuracy_given_attempted": accuracy_given_attempted,
            "f1": f1,
        }

    return metric_calculator
