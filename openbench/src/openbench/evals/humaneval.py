from inspect_ai import Task, task, Epochs
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from openbench.scorers.humaneval import verify
from openbench.datasets.humaneval import get_humaneval_dataset
from typing import Optional


# Adapted from https://github.com/UKGovernmentBEIS/inspect_evals
@task
def humaneval(instruction_prompt: Optional[str] = None) -> Task:
    """
    Inspect Task implementation for the HumanEval benchmark.

    Args:
        instruction_prompt (str, optional): The prompt to prepend to the code problem.
            If None, uses the default HumanEval instruction.

    Returns:
        Task: The configured HumanEval task.
    """
    epochs_count = 5
    reducer_list = ["mean", "pass_at_1", "pass_at_2", "pass_at_5"]

    dataset = (
        get_humaneval_dataset()
        if instruction_prompt is None
        else get_humaneval_dataset(instruction_prompt=instruction_prompt)
    )

    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=verify(),
        sandbox="local",
        config=GenerateConfig(
            temperature=0.5,
        ),
        epochs=Epochs(epochs_count, reducer=reducer_list),
    )
