"""GraphWalks metrics for evaluation scoring."""

from typing import List
from inspect_ai.scorer import (
    Metric,
    Value,
    SampleScore,
    metric,
)

GRAPHWALKS_BINS = [
    (0, 512),
    (512, 1024),
    (1024, 2048),
    (2048, 4096),
    (4096, 8192),
    (8192, 16384),
    (16384, 32768),
    (32768, 65536),
]


@metric
def graphwalks_metrics() -> Metric:
    """Mean F1 by token-count bin (flat mapping)."""

    def metric_calculator(scores: List[SampleScore]) -> Value:
        # output dict
        f1_by_token_count_bin: dict[str, float] = {
            f"{L}-{R}": 0.0 for (L, R) in GRAPHWALKS_BINS
        }

        # internal accumulators
        f1_sums = dict.fromkeys(GRAPHWALKS_BINS, 0.0)
        bin_counts = dict.fromkeys(GRAPHWALKS_BINS, 0)

        if not scores:
            return f1_by_token_count_bin

        for s in scores:
            if s.score.metadata is None:
                continue
            bin_index = s.score.metadata.get("bin_index")
            if (
                not isinstance(bin_index, int)
                or bin_index < 0
                or bin_index >= len(GRAPHWALKS_BINS)
            ):
                continue

            # add individual score and count to running totals
            key = GRAPHWALKS_BINS[bin_index]
            f1_sums[key] += s.score.as_float()
            bin_counts[key] += 1

        # average f1 per bin (divide by count)
        for L, R in GRAPHWALKS_BINS:
            total = f1_sums[(L, R)]
            cnt = bin_counts[(L, R)]
            f1_by_token_count_bin[f"{L}-{R}"] = (total / cnt) if cnt > 0 else 0.0

        return f1_by_token_count_bin

    return metric_calculator


@metric
def graphwalks_token_counts() -> Metric:
    def calc(scores: List[SampleScore]) -> Value:
        counts = {f"{L}-{R}": 0 for (L, R) in GRAPHWALKS_BINS}
        for s in scores:
            if s.score.metadata is None:
                continue
            bin_index = s.score.metadata.get("bin_index")
            if isinstance(bin_index, int) and 0 <= bin_index < len(GRAPHWALKS_BINS):
                L, R = GRAPHWALKS_BINS[bin_index]
                counts[f"{L}-{R}"] += 1
        # flat dict; numeric values
        return {f"samples_per_bin[{k}]": float(v) for k, v in counts.items()}

    return calc
