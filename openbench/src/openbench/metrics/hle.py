"""HLE metrics for evaluation scoring."""

from typing import List
from inspect_ai.scorer import (
    metric,
    Metric,
    Value,
    SampleScore,
)


@metric
def hle_metrics() -> Metric:
    """Calculate HLE specific metrics including average confidence."""

    def metric_calculator(scores: List[SampleScore]) -> Value:
        if not scores:
            return {
                "avg_confidence": 0.0,
            }

        confidences = []

        for sample_score in scores:
            # Get confidence from metadata
            metadata = sample_score.score.metadata
            if metadata and "confidence" in metadata:
                confidences.append(metadata["confidence"])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 100.0

        return {
            "avg_confidence": avg_confidence,
        }

    return metric_calculator
