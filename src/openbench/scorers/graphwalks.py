# src/openbench/scorers/graphwalks.py
from __future__ import annotations

import re
import bisect
from typing import Iterable, Tuple

from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    mean,
    Metric,
    Value,
    SampleScore,
    metric,
)
from openbench.utils.text import get_token_count


def _parse_nodes(response: str) -> tuple[list[str], bool]:
    # get last line of assistant response
    last_line = response.split("\n")[-1]

    # check formatting with case-insensitive matching
    if "final answer:" not in last_line.lower():
        return [], True

    # more flexible regex with case-insensitive flag and end-of-line anchor
    list_part = re.search(r"final answer:\s*\[(.*)\]\s*$", last_line, re.IGNORECASE)
    if list_part:
        inner = list_part.group(1)
        # return [] if empty list (not [""])
        result_list = [item.strip() for item in inner.split(",") if item.strip()]
        # in-order deduplication
        result_list = list(dict.fromkeys(result_list))
        return result_list, False
    else:
        return [], True


def _prf1(pred: Iterable[str], gold: Iterable[str]) -> Tuple[float, float, float]:
    sp, sg = set(pred), set(gold)

    # special case for correct empty set prediction
    if not sp and not sg:
        p, r, f1 = 1.0, 1.0, 1.0
    else:
        n_overlap = len(sp & sg)
        r = n_overlap / len(sg) if sg else 0.0
        p = n_overlap / len(sp) if sp else 0.0
        f1 = 2 * r * p / (r + p) if (r + p) else 0.0

    return p, r, f1


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

    def metric_calculator(scores: list[SampleScore]) -> Value:
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
    def calc(scores: list[SampleScore]) -> Value:
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


@scorer(metrics=[mean(), graphwalks_metrics(), graphwalks_token_counts()])
def graphwalks_scorer():
    async def score(state, target: Target) -> Score:
        # parse prediction + compute PRF1
        pred, parse_err = _parse_nodes(state.output.completion or "")
        gold = list(target)
        p, r, f1 = _prf1(pred, gold)

        # token counts (input + output)
        input_tok_cnt = state.metadata.get("raw_input_tok_cnt", 0)
        output_tok_cnt = get_token_count(state.output.completion or "")
        total_tok_cnt = input_tok_cnt + output_tok_cnt
        state.metadata["total_tok_cnt"] = total_tok_cnt

        # compute bin_index
        bin_boundaries = [right for _, right in GRAPHWALKS_BINS[:-1]]
        bin_index = bisect.bisect_right(bin_boundaries, total_tok_cnt)

        # return per-sample score
        return Score(
            value=float(f1),
            answer=str(pred),
            metadata={
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "parsed_ok": (not parse_err),
                "pred": pred,
                "gold": gold,
                "raw_input_tok_cnt": input_tok_cnt,
                "total_tok_cnt": total_tok_cnt,
                "bin_index": bin_index,
            },
        )

    return score
