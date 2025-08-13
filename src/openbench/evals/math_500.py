from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.math_500 import get_dataset
from openbench.scorers.math_500 import (
    expression_equivalance,
    expression_exact_match_sympy,
)


# Template for solving math problems - from simple-evals
QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()


@task
def math_500(
    grader_model: str = "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
) -> Task:
    """MATH-500: A 500-problem subset of the MATH dataset.

    A smaller, representative subset of the full MATH dataset for faster evaluation.
    Uses the same scoring and configuration as the full dataset.

    Args:
        grader_model: Model to use for checking answer equality (defaults to llama-4-maverick-17b-128e-instruct)

    Returns:
        Task configured for MATH-500 evaluation
    """
    # Get the dataset and format problems
    dataset = get_dataset("math_500_test")
    for sample in dataset:
        sample.input = QUERY_TEMPLATE.format(prompt=sample.input)

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=[
            expression_equivalance(model=grader_model),
            expression_exact_match_sympy(),
        ],
        name="math_500",
        config=GenerateConfig(
            max_tokens=8192,  # Allow long reasoning chains
        ),
    )
