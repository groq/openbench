"""
CRUXEval scorer for OpenBench.

Evaluates model's ability to predict inputs or outputs for given Python functions.
"""

import re
from typing import Callable, Tuple
from inspect_ai.solver import TaskState
from inspect_ai.scorer import (
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
    CORRECT,
    INCORRECT,
)


def parse_structured_response(response: str, task_type: str) -> Tuple[str, str]:
    """
    Parse the structured response to extract the answer and thought process from tags.

    Args:
        response: The model's response containing [ANSWER] tags
        task_type: Either "input" or "output" to determine parsing strategy

    Returns:
        Tuple containing (answer, thought_process)
    """
    # Extract thought process section if present
    thought_match = re.search(r"\[THOUGHT\](.*?)\[/THOUGHT\]", response, re.DOTALL)
    thought_process = thought_match.group(1).strip() if thought_match else ""

    # Extract answer section
    answer_match = re.search(r"\[ANSWER\](.*?)\[/ANSWER\]", response, re.DOTALL)
    if not answer_match:
        return "", thought_process

    answer = answer_match.group(1).strip()

    # Extract the value from the assertion
    if task_type == "input":
        # For input prediction: extract from "assert f(...) == output"
        match = re.search(r"assert f\((.*?)\) == ", answer)
        if match:
            return match.group(1), thought_process
    else:
        # For output prediction: extract from "assert f(input) == ..."
        match = re.search(r"== (.*?)$", answer)
        if match:
            return match.group(1), thought_process

    return "", thought_process


def clean_value(value: str) -> str:
    """
    Clean a value by removing quotes and extra whitespace.

    Args:
        value: The string value to clean

    Returns:
        The cleaned string with quotes and whitespace removed

    Examples:
        >>> clean_value("  'hello'  ")
        'hello'
        >>> clean_value('  "test"')
        'test'
        >>> clean_value('123')
        '123'
    """
    # Remove leading/trailing whitespace
    value = value.strip()

    # Remove matching quotes if present
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1]

    return value.strip()


@scorer(metrics=[accuracy(), stderr()])
def cruxeval_scorer() -> Callable:
    """
    Scorer that evaluates model's ability to predict inputs (CRUXEval-I)
    or outputs (CRUXEval-O) for given Python functions.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get task type from metadata
        task_type = state.metadata.get("task_type", "output")

        # Parse the structured response to get the answer and thought process
        answer, thought_process = parse_structured_response(
            state.output.completion, task_type
        )

        # Get the expected answer and clean both
        expected = clean_value(target.text)
        answer = clean_value(answer)

        # Compare answer with expected answer
        is_correct = answer == expected

        # Build predictions dictionary for metadata
        predictions = {
            "answer": answer,
            "expected": expected,
            "is_correct": is_correct,
            "thought_process": thought_process,
            "task_type": task_type,
        }

        # Return the score
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=answer,
            explanation=f"Expected: {expected}\nGot: {answer}",
            metadata={"predictions": predictions},
        )

    return score
