"""DocVQA (Document Visual Question Answering) evaluation task.

DocVQA evaluates a model's ability to answer questions about document images
including forms, reports, tables, diagrams, and other real-world documents.

This implementation uses model-graded evaluation (LLM-as-judge) to score answers.

Reference: https://arxiv.org/abs/2007.00398
Homepage: https://www.docvqa.org/
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from openbench.datasets.docvqa import get_docvqa_dataset
from openbench.scorers.docvqa import docvqa_model_graded_scorer

# Prompt template for structured answer format
FREEFORM_TEMPLATE = r"""
Answer the following question. The entire content of your response should be of the following format: 'ANSWER: $ANSWER' (without quotes) where $ANSWER is your answer.

{question}
"""


@solver
def docvqa_solver() -> Solver:
    """Solver that wraps questions with structured answer format.

    Applies FREEFORM_TEMPLATE to guide the model to provide answers
    in a consistent format for easier extraction.

    Returns:
        Solver that modifies the user prompt before generation
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = FREEFORM_TEMPLATE.format(
            question=state.user_prompt.text
        )
        return await generate(state)

    return solve


@task
def docvqa(
    split: str = "validation",
) -> Task:
    """DocVQA: Document Visual Question Answering benchmark.

    Evaluates models on answering questions about document images using
    model-graded evaluation (LLM-as-judge).

    Scoring uses GPT-4o-mini as a judge to evaluate whether model answers
    match the ground truth answers, handling semantic equivalence better
    than exact string matching.

    Args:
        split: Dataset split - "validation" (5,349 samples) or "test" (5,188 samples)
               Note: test split has no answers, for leaderboard submission only

    Returns:
        Task configured for DocVQA evaluation with model grading
    """
    # Load dataset with disk-based image caching
    dataset = get_docvqa_dataset(split=split)

    return Task(
        dataset=dataset,
        solver=[
            docvqa_solver(),
        ],
        scorer=docvqa_model_graded_scorer(
            model=get_model(
                "openai/gpt-4o-mini",
                config=GenerateConfig(
                    temperature=0.0,
                    seed=42,
                ),
            ),
        ),
    )
