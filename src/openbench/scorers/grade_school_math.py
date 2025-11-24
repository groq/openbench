"""Base scorer for grade school math problems (GSM8K, MGSM, etc.)."""

from typing import Callable

from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from openbench.utils.text import normalize_number, parse_numeric_answer


async def score_numeric_answer(state: TaskState, target: Target) -> Score:
    """Core scoring logic for numeric answer extraction and comparison."""
    model_output = state.output.completion
    metadata = state.metadata
    answer_prefix = metadata.get("answer_prefix", "Answer")

    extracted_answer = parse_numeric_answer(model_output, answer_prefix)
    normalized_extracted = normalize_number(extracted_answer)
    normalized_target = normalize_number(target.text)

    is_correct = normalized_extracted == normalized_target

    return Score(
        value=1.0 if is_correct else 0.0,
        answer=extracted_answer if extracted_answer else "[No answer found]",
        explanation=f"Extracted: {extracted_answer}, Target: {target.text}, Match: {is_correct}",
        metadata={
            "extracted_answer": extracted_answer,
            "normalized_extracted": normalized_extracted,
            "normalized_target": normalized_target,
        },
    )


@scorer(metrics=[accuracy(), stderr()])
def grade_school_math_scorer() -> Callable:
    """Scorer for grade school math problems using numeric answer extraction."""
    return score_numeric_answer
