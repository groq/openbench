from typing import Callable
from inspect_ai.scorer import scorer, accuracy, stderr, Score, Target, std
from inspect_ai.model import Model, get_model, ChatMessageUser
from inspect_ai.solver import TaskState
from openbench.metrics import grouped


SMT_GRADER_PROMPT = """
You are a mathematics expert tasked with grading Stanford Math Tournament solutions. You will be given:

A student's complete solution with their reasoning
The correct answer

Grade the student solution as either CORRECT or INCORRECT, based on whether the student's final answer matches the correct answer.
Only respond with a single word: either "CORRECT" or "INCORRECT".

Student Solution:
{response}

Correct Answer:
{correct_answer}

Grade (CORRECT/INCORRECT):""".strip()


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        std(),
        grouped(group_key="category", metric=[accuracy(), stderr(), std()]),
    ]
)
def smt_scorer(model: str = "openai/gpt-4.1-mini-2025-04-14") -> Callable:
    """
    SMT scorer using LLM-based grading for mathematical problem evaluation.

    Args:
        model: The model identifier for the grader (default: GPT-4.1-mini)

    Returns:
        A scorer function that evaluates SMT responses
    """
    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        predicted_answer = state.output.completion
        answer = target.text

        grader_prompt = SMT_GRADER_PROMPT.format(
            response=predicted_answer, correct_answer=answer
        )

        message = ChatMessageUser(content=grader_prompt)
        grading_response = await grader_model.generate([message])
        grading_text = grading_response.completion.strip().upper()

        # Explicitly check for "CORRECT" to avoid false positives
        # Only mark as correct if the grader explicitly says "CORRECT"
        is_correct = "CORRECT" in grading_text and "INCORRECT" not in grading_text

        return Score(
            value=1.0 if is_correct else 0.0,
            answer=predicted_answer,
            explanation=f"Grader response: {grading_text}",
        )

    return score
