"""MRCR metrics for evaluation scoring."""

from typing import List
from inspect_ai.scorer import (
    Metric,
    Value,
    SampleScore,
    metric,
)

OPENAI_MRCR_BINS = [
    (4096, 8192),
    (8192, 16384),
    (16384, 32768),
    (32768, 65536),
    (65536, 131072),
    (131072, 262144),
    (262144, 524288),
    (524288, 1048576),
]


@metric
def mrcr_metrics() -> Metric:
    """Calculate MRCR specific metrics: accuracy by token count bin.

    Bin boundaries are:
    [4096, 8192], (8192, 16384], (16384, 32768], (32768, 65536], (65536, 131072], (131072, 262144], (262144, 524288], (524288, 1048576]
    """

    def metric_calculator(scores: List[SampleScore]) -> Value:
        accuracy_by_token_count_bin: dict[str, float] = {}
        bin_counts: dict[str, int] = {}

        for left_bin, right_bin in OPENAI_MRCR_BINS:
            bin_key = f"{left_bin}-{right_bin}"
            accuracy_by_token_count_bin[bin_key] = 0.0
            bin_counts[bin_key] = 0

        if not scores:
            return accuracy_by_token_count_bin

        for sample_score in scores:
            if sample_score.score.metadata is None:
                continue
            bin_index = sample_score.score.metadata.get("bin_index")
            if (
                not isinstance(bin_index, int)
                or bin_index < 0
                or bin_index >= len(OPENAI_MRCR_BINS)
            ):
                continue
            left_bin, right_bin = OPENAI_MRCR_BINS[bin_index]
            bin_key = f"{left_bin}-{right_bin}"
            accuracy_by_token_count_bin[bin_key] += sample_score.score.as_float()
            bin_counts[bin_key] += 1

        # calculate accuracy for each bin
        for bin in accuracy_by_token_count_bin:
            if bin_counts[bin] == 0:
                continue
            accuracy_by_token_count_bin[bin] = (
                accuracy_by_token_count_bin[bin] / bin_counts[bin]
            )

        return accuracy_by_token_count_bin

    return metric_calculator
