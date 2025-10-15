from inspect_ai import Task, task, Epochs
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.mbpp import get_mbpp_dataset
from openbench.scorers.mbpp import verify


@task
def mbpp(subset: str = "full", split: str = "test") -> Task:
    """
    Inspect Task implementation for the MBPP benchmark.

    Args:
        subset: Which subset to evaluate ("full" (default), "sanitized")
        split: Which split to evaluate ("test" (default), "train", "validation", "prompt")
    Returns:
        Task: The configured MBPP task.
    """
    epochs_count = 5
    temperature = 0.5
    return Task(
        name="mbpp",
        dataset=get_mbpp_dataset(subset=subset, split=split),
        solver=generate(),
        scorer=verify(),
        sandbox="local",
        epochs=Epochs(epochs_count, reducer="mean"),
        config=GenerateConfig(temperature=temperature),
    )
