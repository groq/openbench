import json
from inspect_ai.scorer import (
    scorer,
    Score,
    CORRECT,
    INCORRECT,
    Target,
    accuracy,
    stderr,
    std,
)
from inspect_ai.solver import TaskState
from openbench.utils.arc_parsing import parse_arc_response


@scorer(metrics=[accuracy(), stderr(), std()])
def arc_agi_scorer():
    """
    Scorer for ARC-AGI tasks that compares predicted grids with expected outputs.

    Uses exact match: the predicted grid must exactly match the expected grid
    for the sample to be considered correct.
    """

    async def score(state: TaskState, target: Target) -> Score:
        """
        Score a single ARC-AGI sample.

        Args:
            state: TaskState containing model response and sample metadata
            target: Expected output as JSON string

        Returns:
            Score: CORRECT if exact match, INCORRECT otherwise
        """
        # Parse the model response to extract predicted grid
        predicted_grid = parse_arc_response(state.output.completion)

        # Get expected grid by parsing target (JSON string)
        try:
            expected_grid = json.loads(target.text) if target.text else None
        except (json.JSONDecodeError, TypeError):
            return Score(
                value=INCORRECT, explanation="Could not parse expected grid from target"
            )

        if expected_grid is None:
            return Score(
                value=INCORRECT, explanation="No expected grid found in target"
            )

        # Handle parsing failure
        if predicted_grid is None:
            return Score(
                value=INCORRECT,
                explanation="Could not parse predicted grid from model response",
            )

        # Exact match comparison
        if predicted_grid == expected_grid:
            return Score(
                value=CORRECT,
                explanation="Predicted grid matches expected grid exactly",
            )
        else:
            return Score(
                value=INCORRECT,
                explanation=f"Grid mismatch. Expected: {expected_grid}, Got: {predicted_grid}",
            )

    return score
