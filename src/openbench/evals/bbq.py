from inspect_ai import Task, task
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from openbench.datasets.bbq import load_dataset


@task
def bbq(subtask: str | None = None):
    task_dataset = load_dataset(subtask)  # type: ignore

    return Task(
        dataset=task_dataset,
        solver=multiple_choice(),
        scorer=choice(),
    )
