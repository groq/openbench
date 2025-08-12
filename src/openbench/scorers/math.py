import re
from typing import Callable
from inspect_ai.scorer import (
    accuracy,
    scorer,
    stderr,
    Score,
    Target,
)
from inspect_ai.solver import TaskState
from inspect_ai.model import get_model, ChatMessageUser, Model


# Pattern to extract answer from model output
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

# Template for checking mathematical equality
EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: {expr1}
    Expression 2: {expr2}
""".strip()


async def check_equality(grader_model: Model, expr1: str, expr2: str) -> bool:
    """Check if two mathematical expressions are equivalent using model grading."""
    if expr1 is None or expr2 is None:
        return False

    prompt = EQUALITY_TEMPLATE.format(expr1=expr1, expr2=expr2)
    message = ChatMessageUser(content=prompt)

    response = await grader_model.generate([message])
    response_text = response.completion.lower().strip()

    return response_text == "yes"


@scorer(metrics=[accuracy(), stderr()])
def math_scorer(model: str = "openai/gpt-4-turbo-preview") -> Callable:
    """MATH scorer using model-based equality checking.

    Args:
        model: Model to use for checking mathematical equality (defaults to gpt-4-turbo-preview)

    Returns:
        A scorer function for MATH problems
    """
    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        # Extract answer from model output using the pattern
        model_output = state.output.completion
        match = re.search(ANSWER_PATTERN, model_output)
        extracted_answer = match.group(1) if match else None

        # Check equality between extracted answer and target
        if extracted_answer:
            is_correct = await check_equality(
                grader_model, target.text, extracted_answer
            )
        else:
            is_correct = False

        # Return score with metadata
        return Score(
            value=1.0 if is_correct else 0.0,
            answer=extracted_answer,
            metadata={
                "extracted_answer": extracted_answer,
                "target_answer": target.text,
                "model_output": model_output,
            },
        )

    return score
