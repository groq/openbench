from __future__ import annotations

import math

from inspect_ai.scorer import (
    Score,
    ScoreReducer,
    ValueToFloat,
    score_reducer,
    value_to_float,
)
from inspect_ai.scorer._reducer.reducer import (
    _compute_dict_stat,
    _compute_list_stat,
    _compute_scalar_stat,
)
from inspect_ai.scorer._reducer.registry import REDUCER_NAME


@score_reducer(name="pass_hat")
def pass_hat(
    k: int,
    value_to_float_fn: ValueToFloat = value_to_float(),
) -> ScoreReducer:
    """Pass^k calculation, see https://arxiv.org/abs/2406.12045"""

    def reduce(scores):
        if not scores:
            return Score(value=0.0)

        def pass_hat_stat(values: list[float]) -> float:
            if k <= 0:
                return 0.0
            total = len(values)
            if total < k:
                return 0.0
            successes = sum(1 for v in values if v >= 1.0)
            if successes < k:
                return 0.0
            denominator = math.comb(total, k)
            if denominator == 0:
                return 0.0
            return math.comb(successes, k) / denominator

        if isinstance(scores[0].value, dict):
            return _compute_dict_stat(scores, value_to_float_fn, pass_hat_stat)
        elif isinstance(scores[0].value, list):
            return _compute_list_stat(scores, value_to_float_fn, pass_hat_stat)
        else:
            return _compute_scalar_stat(scores, value_to_float_fn, pass_hat_stat)

    setattr(pass_hat, REDUCER_NAME, f"pass_hat_{k}")
    return reduce
