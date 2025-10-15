"""Scorer for BrowseComp evaluation."""

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


# Grading template from the original BrowseComp implementation
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()


# Query template from the original BrowseComp implementation
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()


@scorer(metrics=[accuracy(), stderr()])
def browsecomp_scorer(model: str = "openai/gpt-4.1-2025-04-14") -> Callable:
    """BrowseComp scorer using model grading.

    Args:
        model: Model to use for grading responses (defaults to gpt-4.1-2025-04-14)

    Returns:
        Scorer function for BrowseComp evaluation
    """
    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        # Get the plain question from metadata (not the formatted input)
        # This matches the simple-evals implementation where the grader gets the plain question
        question = state.metadata.get("plain_question", state.input_text)

        # Get the predicted answer from the model output
        predicted_answer = state.output.completion

        # Format the grading prompt
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            correct_answer=target.text,
            response=predicted_answer,
        )

        # Create the message for grading
        message = ChatMessageUser(content=grader_prompt)

        # Get grading response
        grading_response = await grader_model.generate([message])
        grading_text = grading_response.completion

        # Extract whether the answer is correct
        # Look for "correct: yes" or "correct: no" in the response
        match = re.search(r"correct:\s*(yes|no)", grading_text, re.IGNORECASE)
        is_correct = (
            (match.group(1).lower() == "yes") if (match and match.group(1)) else False
        )

        # Extract confidence if available
        confidence_match = re.search(
            r"confidence:\s*(\d+)(?:\s*%)?", grading_text, re.IGNORECASE
        )
        confidence = (
            int(confidence_match.group(1))
            if (confidence_match and confidence_match.group(1))
            else 100
        )  # Default to 100 if not found

        # Return score with metadata
        return Score(
            value=1.0 if is_correct else 0.0,
            answer=predicted_answer,
            metadata={
                "is_correct": is_correct,
                "confidence": confidence,
                "grading_response": grading_text,
            },
        )

    return score
