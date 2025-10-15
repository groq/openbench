from openbench.evals.matharena.matharena import matharena_task
from inspect_ai import Task, task


# Not MathArena, but concatenated from aime_2024_I and aime_2024_II
@task
def aime_2024() -> Task:
    return matharena_task(
        dataset_path="AarushSah/aime2024",
        instruction="Please reason step by step, and put your final answer within \\boxed{{}}.\nThe answer is an integer between 0 and 999 inclusive.",
        default_temperature=0.6,
        default_max_tokens=8000,
        default_epochs=4,
        name="aime_2024",
    )
