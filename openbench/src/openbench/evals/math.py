from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.math import get_dataset
from openbench.scorers.math import math_scorer


# Template for solving math problems - from simple-evals
QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()


@task
def math(grader_model: str = "openai/gpt-4-turbo-preview") -> Task:
    """MATH: Measuring Mathematical Problem Solving.

    Based on the paper by Hendrycks et al. (2021).
    Tests mathematical problem-solving across multiple difficulty levels and topics.
    Uses model-based grading to check mathematical equivalence of answers.

    Args:
        grader_model: Model to use for checking answer equality (defaults to gpt-4-turbo-preview)

    Returns:
        Task configured for MATH evaluation
    """
    # Get the dataset and format problems
    dataset = get_dataset("math_test")
    for sample in dataset:
        sample.input = QUERY_TEMPLATE.format(problem=sample.input)

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=math_scorer(model=grader_model),
        name="math",
        config=GenerateConfig(
            max_tokens=8192,  # Allow long reasoning chains
        ),
    )


@task
def math_500(grader_model: str = "openai/gpt-4-turbo-preview") -> Task:
    """MATH-500: A 500-problem subset of the MATH dataset.

    A smaller, representative subset of the full MATH dataset for faster evaluation.
    Uses the same scoring and configuration as the full dataset.

    Args:
        grader_model: Model to use for checking answer equality (defaults to gpt-4-turbo-preview)

    Returns:
        Task configured for MATH-500 evaluation
    """
    # Get the dataset and format problems
    dataset = get_dataset("math_500_test")
    for sample in dataset:
        sample.input = QUERY_TEMPLATE.format(problem=sample.input)

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=math_scorer(model=grader_model),
        name="math_500",
        config=GenerateConfig(
            max_tokens=8192,  # Allow long reasoning chains
        ),
    )
