from __future__ import annotations

from difflib import SequenceMatcher
from typing import Callable

from inspect_ai.scorer import (
    Score,
    Target,
    scorer,
    mean,
)
from inspect_ai.solver import TaskState

from openbench.utils.text import get_token_count
from openbench.metrics.mrcr import mrcr_metrics, OPENAI_MRCR_BINS


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
