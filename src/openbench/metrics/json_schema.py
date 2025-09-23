"""JSON Schema metrics for evaluation scoring."""

from typing import List
from inspect_ai.scorer import (
    metric,
    Metric,
    Value,
    SampleScore,
)


@metric
def json_validity() -> Metric:
    """Calculates the percentage of successful API calls that produced valid JSON (empirical coverage)."""

    def metric_calculator(scores: List[SampleScore]) -> Value:
        if not scores:
            return 0.0

        # Get samples that had successful API calls (no API errors)
        successful_api_scores = [
            score
            for score in scores
            if score.score.metadata and not score.score.metadata.get("api_error", False)
        ]

        if not successful_api_scores:
            return 0.0

        json_valid_count = sum(
            1
            for score in successful_api_scores
            if score.score.metadata and score.score.metadata.get("json_valid", False)
        )
        return json_valid_count / len(successful_api_scores)

    return metric_calculator


@metric
def schema_compliance() -> Metric:
    """Calculates the percentage of valid JSON outputs that conform to schema."""

    def metric_calculator(scores: List[SampleScore]) -> Value:
        if not scores:
            return 0.0

        valid_json_scores = [
            score
            for score in scores
            if score.score.metadata and score.score.metadata.get("json_valid", False)
        ]

        if not valid_json_scores:
            return 0.0

        schema_compliant_count = sum(
            1
            for score in valid_json_scores
            if score.score.metadata
            and score.score.metadata.get("schema_compliant", False)
        )
        return schema_compliant_count / len(valid_json_scores)

    return metric_calculator


@metric
def api_success_rate() -> Metric:
    """Calculates the percentage of samples that didn't have API errors."""

    # TODO: Change this to only check for structured output related errors
    def metric_calculator(scores: List[SampleScore]) -> Value:
        if not scores:
            return 0.0

        api_success_count = sum(
            1
            for score in scores
            if score.score.metadata and not score.score.metadata.get("api_error", False)
        )
        return api_success_count / len(scores)

    return metric_calculator
