"""MM-Vet v2 scorer using GPT-4 as judge.

Implements the exact few-shot grading prompt from the MM-Vet v2 paper
(arXiv:2408.00765) Table 1, using GPT-4 (gpt-4-0613) to score predictions
on a 0.0-1.0 scale with 0.1 increments.
"""

import re
from typing import Callable

from inspect_ai.model import ChatMessageUser, GenerateConfig, Model, get_model
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr, std
from inspect_ai.solver import TaskState


# Exact few-shot prompt from MM-Vet v2 paper Table 1
MMVET_V2_GRADER_TEMPLATE = """
Compare the ground truth and prediction from AI models, to give a correctness score for the prediction.
<image> in the question indicates where an image is. <AND> in the ground truth means it is totally
right only when all elements in the ground truth are present in the prediction, and <OR> means it is
totally right when any one element in the ground truth is present in the prediction. The correctness
score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete
the last space of the correctness score.

| Question | Ground truth | Prediction | Correctness |
| — | — | — | — |
| What is x in the equation?<image> | -1 <AND> -5 | x = 3 | 0.0 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -1 | 0.5 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -5 | 0.5 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -5 or 5 | 0.5 |
| What is x in the equation?<image> | -1 <AND> -5 | x = -1 or x = -5 | 1.0 |
| Can you explain this meme?<image> | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4 |
| Can you explain this meme?<image> | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0 |
| {question} | {ground_truth} | {prediction} |
""".strip()


def _extract_score(text: str) -> float:
    """Extract correctness score from grader response.

    Follows the original MM-Vet implementation by extracting the first token
    and parsing it as a float score between 0.0 and 1.0.

    Args:
        text: Grader model response text

    Returns:
        Float score between 0.0 and 1.0, defaults to 0.0 if parsing fails
    """
    # Extract first token (matches original MM-Vet implementation)
    first_token = text.split()[0].strip() if text.strip() else ""

    # Try to parse as float
    try:
        score = float(first_token)
        # Ensure score is in valid range [0.0, 1.0]
        if 0.0 <= score <= 1.0:
            return score
    except (ValueError, IndexError):
        pass

    # Fallback: look for score pattern in full text
    pattern = r"\b([01]\.\d)\b"
    match = re.search(pattern, text)
    if match:
        score = float(match.group(1))
        if 0.0 <= score <= 1.0:
            return score

    # Handle edge cases
    if "1.0" in text:
        return 1.0
    if re.search(r"\b0\.0\b", text) or text.strip() == "0":
        return 0.0

    # Default to 0 if no valid score found
    return 0.0


@scorer(metrics=[accuracy(), stderr(), std()])
def mmvetv2_scorer(
    grader_model: str = "openai/gpt-4-0613",
    num_grading_attempts: int = 5,
) -> Callable:
    """MM-Vet v2 scorer using GPT-4 as judge.

    Uses the exact few-shot prompt from the MM-Vet v2 paper to evaluate
    model predictions against ground truth answers. Supports <AND> and <OR>
    logic for answers with multiple acceptable elements.

    Following the paper, each sample is graded multiple times (default 5) and
    the scores are averaged to account for GPT-4 variance even at temperature 0.

    Args:
        grader_model: Model to use for grading (default: gpt-4-0613 per paper)
        num_grading_attempts: Number of times to grade each sample (default: 5 per paper)

    Returns:
        Scorer function compatible with Inspect AI

    Raises:
        ValueError: If num_grading_attempts is less than 1
    """
    # Validate configuration
    if num_grading_attempts < 1:
        raise ValueError(
            f"num_grading_attempts must be at least 1, got {num_grading_attempts}"
        )

    model: Model = get_model(grader_model)

    async def score(state: TaskState, target: Target) -> Score:
        """Score a model prediction using GPT-4 as judge."""
        # Extract metadata
        raw_question = state.metadata.get("raw_question", "")
        ground_truth = target
        prediction = state.output.completion
        capability = state.metadata.get("capability", [])

        # Normalize image markers in question for grader
        # Convert <IMG><image_N> to just <image> for consistency with paper
        question_for_grader = re.sub(r"<IMG><image_\d+>", "<image>", raw_question)

        # Format grading prompt with few-shot examples
        grader_prompt = MMVET_V2_GRADER_TEMPLATE.format(
            question=question_for_grader,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        # Run grading multiple times to account for GPT-4 variance (per paper Section 3.3)
        scores = []
        grading_responses = []

        for attempt in range(num_grading_attempts):
            # Call grader model with low temperature for deterministic scoring
            # Using max_tokens=10 to avoid truncation while keeping responses concise
            message = ChatMessageUser(content=grader_prompt)
            grading_response = await model.generate(
                [message],
                config=GenerateConfig(
                    max_tokens=10,  # Allow room for verbose responses like "The score is 1.0"
                    temperature=0.0,  # Deterministic scoring per paper
                ),
            )

            # Parse score from response
            score_value = _extract_score(grading_response.completion)
            scores.append(score_value)
            grading_responses.append(grading_response.completion)

        # Average scores across all attempts (per paper)
        final_score = sum(scores) / len(scores) if scores else 0.0

        return Score(
            value=final_score,
            answer=prediction,
            explanation=f"GPT-4 grading (avg of {num_grading_attempts} runs): {final_score:.2f}",
            metadata={
                "capability": capability,
                "grading_responses": grading_responses,  # All responses for transparency
                "individual_scores": scores,  # All individual scores
                "ground_truth": ground_truth,
                "score": final_score,
                "score_std": (sum((s - final_score) ** 2 for s in scores) / len(scores))
                ** 0.5
                if len(scores) > 1
                else 0.0,
            },
        )

    return score
