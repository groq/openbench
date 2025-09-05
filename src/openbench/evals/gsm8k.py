from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.gsm8k import get_dataset
from openbench.scorers.score_last_number import score_last_number


# Template for solving math problems - from simple-evals
QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form '####: $ANSWER' (without quotes) where $ANSWER is the final answer to the problem.

{problem}

Remember to put your answer after '####' (without quotes), and you do not need to use a \\boxed command.
""".strip()


@task
def gsm8k() -> Task:
    """GSM8K: Grade School Math 8K

    Based on the paper by Cobbe et al. (2021).
    Tests mathematical problem-solving at grade school level.

    Args:
        grader_model: Model to use for checking answer equality (defaults to gpt-4-turbo-preview)

    Returns:
        Task configured for GSM8K evaluation
    """
    # Get the dataset and format problems
    dataset = get_dataset()
    for sample in dataset:
        sample.input = QUERY_TEMPLATE.format(problem=sample.input)

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=score_last_number(),
        name="gsm8k",
        config=GenerateConfig(
            max_tokens=8192,  # Allow long reasoning chains
        ),
    )
