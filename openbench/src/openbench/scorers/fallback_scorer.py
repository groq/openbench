from typing import Literal
from inspect_ai.scorer import Scorer, Score, CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState
from inspect_ai.scorer import scorer, accuracy, std, stderr


@scorer(metrics=[accuracy(), std(), stderr()])
def fallback_scorer(
    scorers: list[Scorer],
    strategy: Literal["first_correct", "first_answer"] = "first_correct",
) -> Scorer:
    """
    A meta-scorer that tries a list of scorers in sequence based on a strategy.

    Args:
        scorers (list[Scorer]): An ordered list of scorers to try.
        strategy (str): The fallback strategy to use.
            - "first_correct" (default): Returns the score from the first scorer
              that finds a CORRECT answer.
            - "first_answer": Returns the score from the first scorer that
              successfully extracts any answer (CORRECT or INCORRECT).
    """

    async def score(state: TaskState, target: Target) -> Score:
        # This will hold the "best effort" score if no early exit happens.
        # We prioritize a score that has an extracted answer over one that doesn't.
        final_score = None

        for individual_scorer in scorers:
            current_score = await individual_scorer(state, target)

            # Skip if the scorer returned None
            if current_score is None:
                continue

            # Update our best-effort final score.
            # A score with an answer is always better than one without.
            if final_score is None or current_score.answer is not None:
                final_score = current_score

            # --- Check for early exit conditions based on the strategy ---

            # Strategy 1: Stop on the first CORRECT answer.
            if strategy == "first_correct" and current_score.value == CORRECT:
                return current_score

            # Strategy 2: Stop on the first extracted answer (correct or not).
            if strategy == "first_answer" and current_score.answer is not None:
                return current_score

        # If we finished the loop without an early exit, return the best we found.
        return final_score or Score(
            value=INCORRECT,
            explanation="All fallback scorers failed to produce a score.",
        )

    return score
