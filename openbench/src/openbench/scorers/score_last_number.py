from inspect_ai.scorer import (
    Score,
    Scorer,
    scorer,
    CORRECT,
    INCORRECT,
    Target,
    accuracy,
    std,
    stderr,
)
from inspect_ai.solver import TaskState
import re


@scorer(metrics=[accuracy(), std(), stderr()])
def score_last_integer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        matches = re.findall(r"-?\d+", state.output.completion)  # Handles integers only
        if not matches:
            return Score(value=INCORRECT, answer=None)
        answer = matches[-1].strip()
        is_correct = answer == target.text.strip()
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=answer,
            explanation=f"The scorer used is score_last_integer. The answer is {answer} and the target is {target.text}.",
        )

    return score


@scorer(metrics=[accuracy(), std(), stderr()])
def score_last_number() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        matches = re.findall(
            r"-?\d+(?:\.\d+)?", state.output.completion
        )  # Handles ints and floats
        if not matches:
            return Score(value=INCORRECT, answer=None)
        answer = matches[-1].strip()
        is_correct = answer == target.text.strip()
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=answer,
            explanation=f"The scorer used is score_last_number. The answer is {answer} and the target is {target.text}.",
        )

    return score
