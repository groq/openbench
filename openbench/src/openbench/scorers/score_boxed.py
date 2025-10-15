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
def score_boxed() -> Scorer:
    """
    A scorer that evaluates answers enclosed in LaTeX \boxed{} or \fbox{} commands.

    This scorer searches for answers wrapped in either \boxed{} or \fbox{} commands
    in the model's output. If multiple boxed answers are found, it uses the last one.
    The answer is considered correct if it exactly matches the target text after
    stripping whitespace.

    Returns:
        Scorer: A scoring function that returns a Score with:
            - value: CORRECT if the boxed answer matches the target, INCORRECT otherwise
            - answer: The extracted answer if found, None if no boxed answer was found
    """

    async def score(state: TaskState, target: Target) -> Score:
        matches = re.findall(r"\\(?:boxed|fbox)\{([^}]*)\}", state.output.completion)
        if not matches:
            return Score(value=INCORRECT, answer=None)
        answer = matches[-1].strip()
        is_correct = answer == target.text.strip()
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=answer,
            explanation=f"The scorer used is score_boxed. The answer is {answer} and the target is {target.text}.",
        )

    return score
