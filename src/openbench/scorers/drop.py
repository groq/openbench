"""DROP scorer for Inspect AI."""

import re
from typing import Callable

from inspect_ai.scorer import (
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from openbench.metrics.drop import drop_metrics, get_drop_metrics


def extract_answer(response: str) -> str:
    """Extract answer from model response."""
    # Look for "Answer: " pattern
    answer_pattern = r"(?i)Answer\s*:\s*([^\n]+)"
    match = re.search(answer_pattern, response)
    if match:
        return match.group(1).strip()

    # If no explicit answer pattern, return the last line that contains content
    lines = response.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line:
            return line

    return response.strip()


@scorer(metrics=[accuracy(), stderr(), drop_metrics()])
def drop_scorer() -> Callable:
    """DROP scorer using exact match and F1 metrics."""

    async def score(state: TaskState, target: Target) -> Score:
        # Extract the answer from model output
        predicted_answer = extract_answer(state.output.completion)

        # Parse multiple correct answers (separated by |)
        correct_answers = target.text.split("|") if target.text else []

        # Calculate metrics for each possible correct answer and take the max
        max_em = 0.0
        max_f1 = 0.0

        for correct_answer in correct_answers:
            correct_answer = correct_answer.strip()
            if correct_answer:
                em, f1 = get_drop_metrics(predicted_answer, correct_answer)
                max_em = max(max_em, em)
                max_f1 = max(max_f1, f1)

        # Score is 1 if exact match, otherwise use F1/100 as partial credit
        score_value = max_em if max_em == 1.0 else max_f1 / 100.0

        return Score(
            value=score_value,
            answer=predicted_answer,
            metadata={
                "exact_match": max_em,
                "f1": max_f1,
                "predicted_answer": predicted_answer,
                "target_answers": correct_answers,
            },
        )

    return score
