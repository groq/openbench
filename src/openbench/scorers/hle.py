import re
from typing import Callable
from inspect_ai.scorer import (
    accuracy,
    scorer,
    stderr,
    Score,
    Target,
    metric,
    Metric,
    Value,
    SampleScore,
)
from inspect_ai.solver import TaskState
from inspect_ai.model import get_model, ChatMessageUser, Model
from openbench.utils.text import extract_confidence_score


# HLE judge prompt template - using raw string to preserve literal \%
JUDGE_PROMPT = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


def parse_judge_response(judge_response: str) -> tuple[str, str, int]:
    """Parse the judge's response to extract correctness, reasoning, and confidence."""
    # Extract if answer is correct (look for "correct: yes" or "correct: no")
    correct_match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    is_correct = correct_match.group(1).lower() if correct_match else "no"

    # Extract reasoning
    reasoning_match = re.search(
        r"reasoning:\s*(.+?)(?=\n\ncorrect:|$)",
        judge_response,
        re.DOTALL | re.IGNORECASE,
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Extract confidence from judge response
    confidence_match = re.search(r"confidence:\s*(\d+)", judge_response, re.IGNORECASE)
    confidence = int(confidence_match.group(1)) if confidence_match else 100

    return is_correct, reasoning, confidence


@metric
def hle_metrics() -> Metric:
    """Calculate HLE specific metrics including average confidence."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return {
                "avg_confidence": 0.0,
            }

        confidences = []

        for sample_score in scores:
            # Get confidence from metadata
            metadata = sample_score.score.metadata
            if metadata and "confidence" in metadata:
                confidences.append(metadata["confidence"])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 100.0

        return {
            "avg_confidence": avg_confidence,
        }

    return metric_calculator


@scorer(metrics=[accuracy(), stderr(), hle_metrics()])
def hle_scorer(model: str = "openai/o3-mini-2025-01-31") -> Callable:
    """HLE scorer using model grading.

    Args:
        model: Model to use for grading (defaults to o3-mini-2025-01-31 as per HLE repo)
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get the grader model - try default first, fallback if not available
        try:
            grader_model: Model = get_model(model)
        except Exception:
            # Fallback to previous default judge model used in HLE
            try:
                grader_model = get_model("openai/gpt-4o-2024-08-06")
            except Exception:
                # Last resort fallback
                grader_model = get_model("openai/gpt-4o")

        # Get question from input
        question = state.input_text

        # Get the model's response
        model_response = state.output.completion

        # First extract confidence from the original model response
        model_confidence = extract_confidence_score(model_response)

        # Format the judge prompt
        judge_prompt = JUDGE_PROMPT.format(
            question=question, response=model_response, correct_answer=target.text
        )

        # Create message for grading
        message = ChatMessageUser(content=judge_prompt)

        # Get grading response
        grading_response = await grader_model.generate([message])
        judge_text = grading_response.completion

        # Parse the judge's response
        is_correct, reasoning, judge_confidence = parse_judge_response(judge_text)

        # Use model's confidence if judge didn't extract one properly
        final_confidence = (
            model_confidence if judge_confidence == 100 else judge_confidence
        )

        # Determine score value
        score_value = 1.0 if is_correct == "yes" else 0.0

        return Score(
            value=score_value,
            answer=model_response,
            explanation=reasoning,
            metadata={
                "correct": is_correct,
                "confidence": final_confidence,
                "judge_response": judge_text,
                "question_id": state.metadata.get("question_id")
                if state.metadata
                else None,
            },
        )

    return score
