from typing import Callable
from inspect_ai.scorer import scorer, accuracy, stderr, Score, Target
from inspect_ai.model import Model, get_model, ChatMessageUser
from inspect_ai.solver import TaskState
from openbench.utils.text import MOCK_AIME_GRADER_PROMPT


@scorer(metrics=[accuracy(), stderr()])
def otis_mock_aime_scorer(model: str = "openai/gpt-4.1-mini-2025-04-14") -> Callable:
    """
    MockAIME scorer using LLM-based grading for mathematical problem evaluation.

    Args:
        model: The model identifier for the grader (default: GPT-4.1-mini)

    Returns:
        A scorer function that evaluates MockAIME responses
    """
    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        predicted_answer = state.output.completion
        answer = target.text

        grader_prompt = MOCK_AIME_GRADER_PROMPT.format(
            response=predicted_answer, correct_solution=answer
        )

        message = ChatMessageUser(content=grader_prompt)
        grading_response = await grader_model.generate([message])
        grading_text = grading_response.completion.strip().upper()

        is_correct = "INCORRECT" not in grading_text

        return Score(
            value=1.0 if is_correct else 0.0,
            answer=predicted_answer,
            explanation=f"Grader response: {grading_text}",
        )

    return score
