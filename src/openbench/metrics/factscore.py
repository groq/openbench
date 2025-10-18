"""Aggregate metrics for the FActScore benchmark."""

from __future__ import annotations

from statistics import mean
from typing import List

from inspect_ai.scorer import Metric, SampleScore, Value, metric


def _mean(values: List[float]) -> float:
    filtered = [v for v in values if v is not None]
    return mean(filtered) if filtered else 0.0


@metric
def factscore_metrics() -> Metric:
    """Compute dataset-level summaries for FActScore."""

    def calculate(scores: List[SampleScore]) -> Value:
        if not scores:
            return {
                "factscore": 0.0,
                "init_score": 0.0,
                "respond_ratio": 0.0,
                "facts_per_response": 0.0,
            }

        responded = 0
        factscore_vals: list[float] = []
        init_vals: list[float] = []
        facts_counts: list[float] = []

        for sample_score in scores:
            metadata = sample_score.score.metadata or {}
            if metadata.get("responded"):
                responded += 1
            factscore_vals.append(metadata.get("factscore", sample_score.score.value))
            init_val = metadata.get("init_score")
            if init_val is not None:
                init_vals.append(init_val)
            facts_count = metadata.get("facts_per_response")
            if facts_count is not None:
                facts_counts.append(facts_count)

        total = len(scores)
        respond_ratio = responded / total if total else 0.0

        return {
            "factscore": _mean(factscore_vals),
            "init_score": _mean(init_vals) if init_vals else 0.0,
            "respond_ratio": respond_ratio,
            "facts_per_response": _mean(facts_counts) if facts_counts else 0.0,
        }

    return calculate
