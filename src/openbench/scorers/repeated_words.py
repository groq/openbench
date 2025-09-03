from typing import Callable
import Levenshtein
from inspect_ai.scorer import (
    scorer,
    mean,
    Score,
    Target,
    metric,
    Metric,
    Value,
    SampleScore,
)
from inspect_ai.solver import TaskState


def normalized_levenshtein_score(target: str, output: str) -> float:
    """Calculate normalized Levenshtein score.

    Args:
        target: The target string
        output: The generated string

    Returns:
        Float between 0 and 1, where 1 is perfect match
    """
    if not target or not output:
        return 0.0

    distance = Levenshtein.distance(target, output)
    max_len = max(len(target), len(output))
    return 1 - (distance / max_len)


@metric
def repeated_words_metrics() -> Metric:
    """Calculate repeated words metrics by number of words (defined by ID)."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return {}

        bins: dict[str, list[float]] = {}
        for sample_score in scores:
            metadata = sample_score.score.metadata
            if not metadata:
                continue

            sample_id = metadata.get("id", "")
            if not sample_id or "_" not in sample_id:
                continue

            word_count_bin = sample_id.split("_")[0]
            if word_count_bin not in bins:
                bins[word_count_bin] = []

            bins[word_count_bin].append(sample_score.score.as_float())

        metrics_by_bin: dict[str, float] = {}
        for bin_name, scores_list in bins.items():
            mean_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
            metrics_by_bin[f"levenshtein_score_{bin_name}"] = mean_score

        return metrics_by_bin

    return metric_calculator


@scorer(metrics=[mean(), repeated_words_metrics()])
def repeated_words_scorer() -> Callable:
    """Scorer for repeated words task using normalized Levenshtein distance.

    Computes the Levenshtein similarity between the target and predicted text,
    normalized by the maximum length of the two strings. Also bins results
    by the number of words (extracted from the sample ID).

    Returns:
        Scorer function that computes Levenshtein similarity scores
    """

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion or ""
        answer = target.text or ""
        sample_id = state.metadata.get("id")

        score_value = normalized_levenshtein_score(answer, response)

        return Score(
            value=score_value,
            answer=response,
            explanation=None,
            metadata={
                "id": sample_id,
                "target": answer,
                "levenshtein_score": score_value,
            },
        )

    return score
