from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.rootly_terraform import load_rootly_terraform_dataset
from openbench.scorers.mcq import simple_mcq_scorer


@task
def rootly_terraform(subtask: str = None) -> Task:  # type: ignore
    dataset = load_rootly_terraform_dataset(subtask)
    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=simple_mcq_scorer(),
        config=GenerateConfig(),
    )
