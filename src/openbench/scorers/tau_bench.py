"""
Scoring logic for tau-bench: rely on tau2's official evaluator output.
"""

from __future__ import annotations

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState


@scorer(metrics=[mean()])
def tau_bench_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        tau2_info = state.metadata.get("tau2")
        if not tau2_info:
            return Score(
                value=0.0,
                explanation="tau2 metadata missing from solver output",
            )
        reward_info = tau2_info.get("reward_info", {})
        reward = float(reward_info.get("reward", 0.0))
        return Score(
            value=reward,
            metadata=tau2_info,
        )

    return score
