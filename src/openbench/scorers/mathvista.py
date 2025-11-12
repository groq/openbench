"""MathVista scorer with answer extraction and comparison logic.

Implements scoring faithful to the original MathVista implementation:
https://github.com/lupantech/MathVista
"""

from typing import Any, Callable, List, Optional

import re

from inspect_ai.model import (
    ChatMessageUser,
    ContentText,
    GenerateConfig,
    get_model,
    Model,
)
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr, std
from inspect_ai.solver import TaskState


async def extract_with_llm(
    response: str,
    question: str,
    question_type: str,
    answer_type: str,
    grader_model: Model,
    choices: Optional[List[str]] = None,
) -> str:
    """Use LLM to extract answer from complex responses.

    Faithful to original MathVista Stage 4 fallback using GPT-4.
    This is used when simpler extraction methods fail.

    Args:
        response: Model's text response
        question: Original question text
        question_type: "multi_choice" or "free_form"
        answer_type: "text", "integer", or "float"
        grader_model: Model to use for extraction
        choices: List of choices for MCQ

    Returns:
        Extracted answer string
    """
    # Build prompt following original MathVista pattern
    prompt_parts = [
        "Please extract the final answer from the following response.",
        f"Question type: {question_type}",
        f"Answer type: {answer_type}",
    ]

    if question_type == "multi_choice" and choices:
        prompt_parts.append(f"Choices: {', '.join(choices)}")

    prompt_parts.extend(
        [
            f"\nQuestion: {question}",
            f"\nModel response: {response}",
            "\nExtract ONLY the final answer without any explanation.",
            "For multiple choice, return the exact choice text.",
            "For numbers, return just the number.",
            "For text, return just the answer word or phrase.",
        ]
    )

    extraction_prompt = "\n".join(prompt_parts)

    # Call the grader model for extraction
    result = await grader_model.generate(
        input=[ChatMessageUser(content=[ContentText(text=extraction_prompt)])],
        # Use conservative settings for extraction
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=100,
        ),
    )

    extracted = result.completion.strip()
    return extracted


async def extract_answer(
    response: str,
    question_type: str,
    answer_type: str,
    choices: Optional[List[str]] = None,
    question: Optional[str] = None,
    use_llm_fallback: bool = True,
    grader_model: Optional[Model] = None,
) -> Any:
    """Extract answer from model response using multi-stage approach.

    Faithful to original MathVista implementation:
    Stage 1: Direct matching for multiple choice
    Stage 2: Type-based conversion for numeric answers
    Stage 3: Pattern-based extraction
    Stage 4: LLM-based extraction (fallback)

    Args:
        response: Model's text response
        question_type: "multi_choice" or "free_form"
        answer_type: "text", "integer", or "float"
        choices: List of choices for MCQ
        question: Original question (for LLM fallback)
        use_llm_fallback: Whether to use LLM extraction as fallback
        grader_model: Model to use for extraction (required if use_llm_fallback=True)

    Returns:
        Extracted answer in appropriate type
    """
    # Stage 1: Direct matching for multiple choice
    if question_type == "multi_choice" and choices:
        for choice in choices:
            if str(choice).lower() == response.lower():
                return choice

    # Stage 2: Type-based conversion for numeric answers
    if answer_type == "integer":
        try:
            # Extract last number from response
            numbers = re.findall(
                r"-?\d+(?:,\d{3})*(?:\.\d+)?", response.replace(",", "")
            )
            if numbers:
                # Use int(float()) to handle "5.0" -> 5
                return int(float(numbers[-1]))
        except (ValueError, IndexError):
            pass

    if answer_type == "float":
        try:
            # Extract last number from response
            numbers = re.findall(
                r"-?\d+(?:,\d{3})*(?:\.\d+)?", response.replace(",", "")
            )
            if numbers:
                return float(numbers[-1])
        except (ValueError, IndexError):
            pass

    # Stage 3: Quick extraction - look for common answer patterns
    # Pattern: "The answer is X" or "Answer: X"
    answer_patterns = [
        r"[Aa]nswer\s*:\s*([A-Z])",  # Answer: A
        r"[Aa]nswer\s*is\s*\(?([A-Z])\)?",  # answer is (A) or answer is A
        r"\(?([A-Z])\)",  # (A) or (B)
        r"^([A-Z])[\.\)]",  # A. or A)
        r"option\s*([A-Z])",  # option A
    ]

    # For MCQ, extract letter
    if question_type == "multi_choice":
        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                letter = match.group(1).upper()
                # Map letter to choice index
                if choices:
                    idx = ord(letter) - ord("A")
                    if 0 <= idx < len(choices):
                        return choices[idx]
                return letter

        # Fallback: Use Levenshtein distance to find closest choice
        if choices:
            from difflib import get_close_matches

            matches = get_close_matches(
                response.lower(), [c.lower() for c in choices], n=1, cutoff=0.6
            )
            if matches:
                idx = [c.lower() for c in choices].index(matches[0])
                return choices[idx]

    # Stage 4: For free-form, return the cleaned response or extracted value
    # Try to extract value after common indicators
    indicators = ["answer is", "answer:", "result is", "result:", "=", "equals"]
    for indicator in indicators:
        if indicator in response.lower():
            parts = response.lower().split(indicator)
            if len(parts) > 1:
                candidate = parts[-1].strip()
                # Try to extract number if answer_type is numeric
                if answer_type in ["integer", "float"]:
                    numbers = re.findall(r"-?\d+(?:\.\d+)?", candidate.replace(",", ""))
                    if numbers:
                        try:
                            if answer_type == "integer":
                                return int(float(numbers[0]))
                            else:
                                return float(numbers[0])
                        except ValueError:
                            pass
                # Return first non-empty token
                tokens = candidate.split()
                if tokens:
                    return tokens[0].strip(".,!?;:")

    # Final fallback: return last line or full response
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    if lines:
        last_line = lines[-1]
        # Try to extract a clean answer from last line
        if answer_type in ["integer", "float"]:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", last_line.replace(",", ""))
            if numbers:
                try:
                    if answer_type == "integer":
                        return int(float(numbers[-1]))
                    else:
                        return float(numbers[-1])
                except ValueError:
                    pass

        # Stage 4: LLM-based extraction
        # Only use for text answers or when other methods failed
        if use_llm_fallback and question and answer_type == "text" and grader_model:
            try:
                llm_extracted = await extract_with_llm(
                    response=response,
                    question=question,
                    question_type=question_type,
                    answer_type=answer_type,
                    grader_model=grader_model,
                    choices=choices,
                )
                return llm_extracted
            except Exception:
                # If LLM extraction fails, continue to final fallback
                pass

        return last_line

    return response


def normalize_answer(
    answer: Any,
    answer_type: str,
    precision: Optional[float] = None,
) -> Any:
    """Normalize answer for comparison.

    Args:
        answer: Answer to normalize
        answer_type: Type of answer ("text", "integer", "float")
        precision: Precision for float rounding (from original dataset)

    Returns:
        Normalized answer
    """
    if answer is None:
        return ""

    # Handle integer type
    if answer_type == "integer":
        try:
            # Convert to int via float to handle "5.0" -> 5
            return int(float(str(answer).replace(",", "")))
        except (ValueError, TypeError):
            return str(answer)

    # Handle float type with precision
    if answer_type == "float":
        try:
            value = float(str(answer).replace(",", ""))
            if precision is not None:
                # Round to specified precision
                return round(value, int(precision))
            return value
        except (ValueError, TypeError):
            return str(answer)

    # Handle list/array type
    if isinstance(answer, (list, tuple)):
        return [normalize_answer(a, answer_type, precision) for a in answer]

    # Default: string normalization
    return str(answer).strip().lower()


def safe_equal(pred: Any, gold: Any) -> bool:
    """Safely compare two answers with type handling.

    Args:
        pred: Predicted answer
        gold: Gold/target answer

    Returns:
        True if answers match, False otherwise
    """
    try:
        # Direct equality
        if pred == gold:
            return True

        # Try string comparison
        if str(pred).strip().lower() == str(gold).strip().lower():
            return True

        # Try numeric comparison with tolerance
        try:
            pred_num = float(str(pred).replace(",", ""))
            gold_num = float(str(gold).replace(",", ""))
            # Use small tolerance for floating point comparison
            return abs(pred_num - gold_num) < 1e-6
        except (ValueError, TypeError):
            pass

        return False
    except Exception:
        return False


def mathvista_scorer(
    grader_model: str = "openai/gpt-4-turbo",
) -> Callable[[], Any]:
    """Create a MathVista scorer with global accuracy metrics.

    Args:
        grader_model: Model to use for LLM-based extraction fallback
                     (default: gpt-4-turbo, matching the original paper's use of GPT-4)

    Returns:
        Scorer function
    """
    grader: Model = get_model(grader_model)

    @scorer(metrics=[accuracy(), stderr(), std()])
    def mathvista_scorer_impl():
        async def score(state: TaskState, target: Target) -> Score:
            """Score a MathVista sample.

            Args:
                state: Task state with model response
                target: Target answer

            Returns:
                Score object (1.0 for correct, 0.0 for incorrect)
            """
            # Extract metadata
            question_type = state.metadata.get("question_type", "free_form")
            answer_type = state.metadata.get("answer_type", "text")
            precision = state.metadata.get("precision")
            choices = state.metadata.get("choices")
            question = state.metadata.get("question", "")

            # Get model response
            model_response = state.output.completion

            # Extract answer from response (async with LLM fallback)
            extracted_answer = await extract_answer(
                response=model_response,
                question_type=question_type,
                answer_type=answer_type,
                choices=choices,
                question=question,
                use_llm_fallback=True,
                grader_model=grader,
            )

            # Normalize both answers
            normalized_pred = normalize_answer(extracted_answer, answer_type, precision)
            normalized_gold = normalize_answer(target.text, answer_type, precision)

            # Store extraction details in metadata for debugging
            state.metadata["extracted_answer"] = str(extracted_answer)
            state.metadata["normalized_pred"] = str(normalized_pred)
            state.metadata["normalized_gold"] = str(normalized_gold)

            # Compare answers
            correct = safe_equal(normalized_pred, normalized_gold)

            return Score(
                value=1.0 if correct else 0.0,
                answer=model_response,
                explanation=f"Predicted: {normalized_pred}, Gold: {normalized_gold}",
            )

        return score

    return mathvista_scorer_impl
