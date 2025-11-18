"""DocVQA model-graded scorer using LLM-as-judge.

This scorer uses GPT-4o-mini to evaluate whether model answers match ground truth.
"""

import re
from typing import Callable

from inspect_ai.model import Model, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.scorer._metric import INCORRECT
from inspect_ai.scorer._metrics import accuracy, stderr
from inspect_ai.solver._task_state import TaskState

# Default template for model-graded evaluation
DEFAULT_TEMPLATE = """
You are evaluating a submission based on the following:

[Question]: {question}
[Answer]:   {answer}
[Criterion]: {criterion}

Does the submission meet the criterion?

{instructions}
""".strip()

# Default grading instructions
DEFAULT_INSTRUCTIONS = """
When you have finished evaluating, provide your answer in the format:

GRADE: C  for a correct answer
GRADE: P  for a partially correct answer (if desired)
GRADE: I  for an incorrect answer

Use step-by-step reasoning, then finish with "GRADE: X" where X is one of the letters above.
""".strip()

# Regex to capture exactly 'C', 'P', or 'I' from "GRADE: C"
DEFAULT_GRADE_PATTERN = r"(?i)GRADE\s*:\s*([CPI])(.*)$"

# Template specific to DocVQA fact-checking
MODEL_GRADED_FACT_TEMPLATE = r"""
You are comparing a submitted answer to an expert answer on a given question. Here is the data:

[BEGIN DATA]
************
[Expert]: {criterion}
************
[Submission]: {answer}
************
[END DATA]

The correct answer is the expert response. Please compare the two answers and determine if the submission is correct. Sometimes, the expert answer may have multiple correct answers, and the submission may have one of them. In this case, the submission is correct. Use your best judgement to determine if the submission is correct.

{instructions}
"""


def _default_chat_history(state: TaskState) -> str:
    """
    Simple history function that concatenates user/assistant messages.

    Args:
        state: Current task state

    Returns:
        Formatted chat history string
    """
    lines = []
    for msg in state.messages:
        if msg.role == "user":
            lines.append(f"User: {msg.text}")
        elif msg.role == "assistant":
            lines.append(f"Assistant: {msg.text}")
    return "\n".join(lines)


@scorer(metrics=[accuracy(), stderr()])
def docvqa_model_graded_scorer(
    template: str | None = None,
    instructions: str | None = None,
    grade_pattern: str | None = None,
    include_history: bool | Callable[[TaskState], str] = False,
    model: str | Model | None = None,
) -> Scorer:
    """Model-graded scorer for DocVQA using LLM-as-judge.

    Uses target.target (all acceptable answers) as the criterion for evaluation.
    An LLM (default: GPT-4o-mini) evaluates whether the model's answer matches
    any of the ground truth answers.


    Args:
        template: Prompt template for grading. Must include variables:
            {question}, {answer}, {criterion}, and {instructions}.
        instructions: Additional instructions for the grading LLM.
        grade_pattern: Regex to extract the grade (C, P, or I).
        include_history: If True, incorporate chat history into question.
            If a callable, it should take TaskState and return a string.
        model: The model or model name to use for grading.
            Defaults to "openai/gpt-4o-mini" with temperature=0.0, seed=42.

    Returns:
        Scorer function compatible with Inspect AI evaluation framework
    """
    final_template = template or MODEL_GRADED_FACT_TEMPLATE
    final_instructions = instructions or DEFAULT_INSTRUCTIONS
    final_grade_pattern = grade_pattern or DEFAULT_GRADE_PATTERN

    async def score(state: TaskState, target: Target) -> Score:
        """Score a single model output against ground truth answers.

        Args:
            state: TaskState containing model output and metadata
            target: Target containing ground truth answer(s)

        Returns:
            Score object with value 'C', 'P', or 'I' and grading explanation
        """
        # Resolve the grading model
        chosen_model = model if isinstance(model, Model) else get_model(model)

        # Present the question
        if include_history is True:
            question = _default_chat_history(state)
        elif callable(include_history):
            question = include_history(state)
        else:
            question = state.input_text

        # Prepare grading prompt
        # Note: using target.target (list of acceptable answers) as criterion
        prompt = final_template.format(
            question=question,
            answer=state.output.completion,
            criterion=target.target,
            instructions=final_instructions,
        )

        # Call the grading model
        result = await chosen_model.generate(prompt)

        # Try to parse the grade from the model's completion
        match = re.search(final_grade_pattern, result.completion)
        if match:
            grade_letter = match.group(1).upper()
            return Score(
                value=grade_letter,
                answer=state.output.completion,
                explanation=result.completion,
                metadata={
                    "prompt": prompt,
                    "grading_model_output": result.completion,
                },
            )
        else:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation="No valid grade found in model output:\n"
                + result.completion,
                metadata={
                    "prompt": prompt,
                    "grading_model_output": result.completion,
                },
            )

    return score
