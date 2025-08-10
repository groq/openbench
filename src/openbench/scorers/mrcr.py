from __future__ import annotations

from difflib import SequenceMatcher
from typing import Callable

from inspect_ai.scorer import (
    Score,
    Target,
    scorer,
    mean,
    Metric,
    Value,
    SampleScore,
    metric,
)
from inspect_ai.solver import TaskState

from openbench.utils.text import get_token_count

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


def _sequence_ratio(
    response: str, answer: str, random_string_to_prepend: str | None
) -> float:
    """Compute SequenceMatcher ratio with MRCR's prefix handling.

    If a random prefix is provided, the ratio is computed after removing the
    prefix from both strings. If the response does not start with the prefix,
    the ratio is 0, matching the reference implementation behavior.
    """
    if (
        not isinstance(random_string_to_prepend, str)
        or len(random_string_to_prepend) == 0
    ):
        return float(SequenceMatcher(None, response, answer).ratio())

    if not response.startswith(random_string_to_prepend):
        return 0.0

    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


@metric
def mrcr_metrics() -> Metric:
    """Calculate MRCR specific metrics: accuracy by token count bin.

    Bin boundaries are:
    [4096, 8192], (8192, 16384], (16384, 32768], (32768, 65536], (65536, 131072], (131072, 262144], (262144, 524288], (524288, 1048576]
    """

    def metric_calculator(scores: list[SampleScore]) -> Value:
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


@scorer(metrics=[mean(), mrcr_metrics()])
def mrcr_scorer() -> Callable:
    """Scorer for MRCR.

    Produces two values in the returned score:
    - value: CORRECT or INCORRECT depending on exact string equality of the
      model response and the target answer.
    - metadata.sequence_ratio: SequenceMatcher ratio computed after handling the
      random prefix as in the reference implementation.

    Args:
        None
    """

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion or ""
        answer = target.text

        prefix = (
            state.metadata.get("random_string_to_prepend") if state.metadata else None
        )
        ratio = _sequence_ratio(
            response=response, answer=answer, random_string_to_prepend=prefix
        )

        # get token count of input and target
        input_tok_cnt = state.metadata.get("raw_input_tok_cnt", 0)
        output_tok_cnt = get_token_count(target.text)
        total_tok_cnt = input_tok_cnt + output_tok_cnt
        state.metadata["total_tok_cnt"] = total_tok_cnt

        # get bin index
        bin_index = 0
        for i, (left_bin, right_bin) in enumerate(OPENAI_MRCR_BINS):
            if i == 0 or i == len(OPENAI_MRCR_BINS) - 1:
                if left_bin <= total_tok_cnt <= right_bin:
                    bin_index = i
                    break
            else:
                if left_bin <= total_tok_cnt < right_bin:
                    bin_index = i
                    break

        return Score(
            value=ratio,
            answer=response,
            explanation=None,
            metadata={"bin_index": bin_index},
        )

    return score
