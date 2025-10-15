"""MGSM scorer for evaluating math problem solutions."""

from typing import Callable
from inspect_ai.scorer import (
    accuracy,
    scorer,
    stderr,
    Score,
    Target,
)
from inspect_ai.solver import TaskState
from openbench.utils.text import parse_numeric_answer, normalize_number
from openbench.metrics.mgsm import language_accuracy


@scorer(metrics=[accuracy(), stderr(), language_accuracy()])
def mgsm_scorer() -> Callable:
    """MGSM scorer for evaluating math problem solutions."""

    async def score(state: TaskState, target: Target) -> Score:
        # Get the model's response
        model_output = state.output.completion

        # Get metadata from the sample
        metadata = state.metadata
        answer_prefix = metadata.get("answer_prefix", "Answer")
        language = metadata.get("language", "en")

        # Extract answer from model output
        extracted_answer = parse_numeric_answer(model_output, answer_prefix)

        # Normalize both extracted answer and target for comparison
        normalized_extracted = normalize_number(extracted_answer)
        normalized_target = normalize_number(target.text)

        # Score is 1.0 if they match, 0.0 otherwise
        is_correct = normalized_extracted == normalized_target
        score_value = 1.0 if is_correct else 0.0

        return Score(
            value=score_value,
            answer=extracted_answer if extracted_answer else "[No answer found]",
            explanation=f"Extracted: {extracted_answer}, Target: {target.text}, Normalized match: {is_correct}",
            metadata={
                "language": language,
                "extracted_answer": extracted_answer,
                "normalized_extracted": normalized_extracted,
                "normalized_target": normalized_target,
            },
        )

    return score
