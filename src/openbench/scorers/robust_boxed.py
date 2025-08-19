"""Enhanced boxed answer extraction scorer with better fallback logic."""

# Adapted from https://github.com/openai/gpt-oss
import re
from typing import Optional

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


def extract_boxed_answer(
    text: str, fallback_to_last_number: bool = True
) -> Optional[str]:
    """
    Extract answer from LaTeX boxed format with optional fallback.

    Searches for answers in \boxed{}, \fbox{}, or \framebox{} commands. If no boxed
    answer is found and fallback is enabled, returns the last number in the text.


    Args:
        text: The text to search for boxed answers
        fallback_to_last_number: Whether to fall back to last number if no box found

    Returns:
        The extracted answer string, or None if not found
    """
    # Look for boxed, fbox, or framebox patterns
    pattern = r"\\(?:boxed|fbox|framebox)\{([^}]*?)\}"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Get the last boxed answer (most likely to be final answer)
        answer = matches[-1]
        # If there are nested braces, extract innermost content
        if "," in answer:
            # Sometimes answers have extra formatting, take last part
            answer = answer.split(",")[-1]
        return answer.strip()

    # Fallback to last number if enabled
    if fallback_to_last_number:
        # Find all numbers (including negative)
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]

    return None


def normalize_numeric_answer(answer: str) -> Optional[str]:
    """
    Normalize a numeric answer for comparison.

    Handles various number formats including:
    - Removing commas
    - Extracting leading integers
    - Removing trailing zeros after decimal


    Args:
        answer: The answer string to normalize

    Returns:
        Normalized answer string, or None if not a valid number
    """
    if not answer:
        return None

    # Remove commas and extra whitespace
    answer = answer.replace(",", "").strip()

    # Try to extract integer from start (for AIME-style answers)
    match = re.match(r"^-?\d+", answer)
    if match:
        return match.group(0)

    # Try to parse as float and normalize
    try:
        num = float(answer)
        # If it's a whole number, return as integer
        if num == int(num):
            return str(int(num))
        # Otherwise remove trailing zeros
        return str(num).rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        return None


@scorer(metrics=[accuracy(), std(), stderr()])
def robust_boxed_scorer(
    fallback_to_last_number: bool = True, normalize_numbers: bool = True
) -> Scorer:
    """
    Enhanced scorer for LaTeX boxed answers with intelligent fallbacks.

    This scorer extracts answers from \boxed{} or \framebox{} commands with
    optional fallback to the last number in the response. It also handles
    number normalization for more robust comparison.

    Args:
        fallback_to_last_number: Whether to use last number if no boxed answer
        normalize_numbers: Whether to normalize numeric answers before comparison

    Returns:
        Scorer: A scoring function with enhanced answer extraction
    """

    async def score(state: TaskState, target: Target) -> Score:
        extracted = extract_boxed_answer(
            state.output.completion, fallback_to_last_number=fallback_to_last_number
        )

        if extracted is None:
            return Score(
                value=INCORRECT,
                answer=None,
                explanation="No boxed answer or number found in response",
            )

        # Normalize if requested
        if normalize_numbers:
            extracted_norm = normalize_numeric_answer(extracted)
            target_norm = normalize_numeric_answer(target.text.strip())

            if extracted_norm is not None and target_norm is not None:
                is_correct = extracted_norm == target_norm
            else:
                # Fall back to string comparison if normalization fails
                is_correct = extracted.strip() == target.text.strip()
        else:
            is_correct = extracted.strip() == target.text.strip()

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=extracted,
            explanation=f"Extracted '{extracted}' from response, target was '{target.text}'",
        )

    return score


@scorer(metrics=[accuracy(), std(), stderr()])
def aime_scorer() -> Scorer:
    """
    Specialized scorer for AIME math competition problems.

    AIME answers are always integers between 0 and 999. This scorer:
    1. Tries to extract from \boxed{} or \framebox{}
    2. Falls back to last integer if no boxed answer
    3. Validates the answer is in valid AIME range
    4. Compares as integers


    Returns:
        Scorer: A scoring function optimized for AIME problems
    """

    async def score(state: TaskState, target: Target) -> Score:
        extracted = extract_boxed_answer(
            state.output.completion, fallback_to_last_number=True
        )

        if extracted is None:
            return Score(
                value=INCORRECT, answer=None, explanation="No answer found in response"
            )

        # Try to parse as integer (AIME answers are always integers)
        try:
            # Extract just the integer part
            match = re.match(r"^-?\d+", extracted.replace(",", "").strip())
            if match:
                answer_int = int(match.group(0))
            else:
                answer_int = int(float(extracted))

            # Validate AIME range
            if not (0 <= answer_int <= 999):
                return Score(
                    value=INCORRECT,
                    answer=str(answer_int),
                    explanation=f"Answer {answer_int} outside valid AIME range [0, 999]",
                )

            # Compare as integers
            target_int = int(target.text.strip())
            is_correct = answer_int == target_int

            return Score(
                value=CORRECT if is_correct else INCORRECT,
                answer=str(answer_int),
                explanation=f"Extracted {answer_int} from response, target was {target_int}",
            )

        except (ValueError, TypeError) as e:
            return Score(
                value=INCORRECT,
                answer=extracted,
                explanation=f"Could not parse '{extracted}' as integer: {e}",
            )

    return score
