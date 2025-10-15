"""Instruction Following metrics implementation."""

from math import sqrt
from inspect_ai.scorer import Metric, SampleScore, Value, metric


def _accuracy_and_stderr(values: list[int]) -> tuple[float, float]:
    total = len(values)
    mean = sum(values) / total
    return mean, (sqrt(mean * (1 - mean) / total) if values else 0.0)


@metric
def ifeval_metrics() -> Metric:
    """Prompt and instruction level accuracies with stderr for strict and loose scoring."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        aggregates: dict[str, dict[str, list[int]]] = {
            "strict": {"instruction": [], "prompt": []},
            "loose": {"instruction": [], "prompt": []},
        }

        for sample_score in scores:
            metadata = sample_score.score.metadata or {}

            for mode, key in (
                ("strict", "strict_follow_instruction_list"),
                ("loose", "loose_follow_instruction_list"),
            ):
                follow_list = metadata[key]
                aggregates[mode]["instruction"].extend(map(int, follow_list))
                aggregates[mode]["prompt"].append(int(all(follow_list)))

        result = {}

        for mode, levels in aggregates.items():
            for level, values in levels.items():
                accuracy, stderr = _accuracy_and_stderr(values)
                result[f"{mode}_{level}_accuracy"] = accuracy
                result[f"{mode}_{level}_stderr"] = stderr

        return result

    return metric_calculator
