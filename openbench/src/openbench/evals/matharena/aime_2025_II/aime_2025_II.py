from openbench.evals.matharena.matharena import matharena_task
from inspect_ai import Task, task


# instruction: "Please reason step by step, and put your final answer within \\boxed{{}}.\nThe answer is an integer between 0 and 999 inclusive."
# default_temperature: 0.6
# default_max_tokens: 8000
# strict_parsing: false
# n_problems: 15
# date: "2025-02-06"
# dataset_path: MathArena/aime_2025_II
@task
def aime_2025_II() -> Task:
    return matharena_task(
        dataset_path="MathArena/aime_2025_II",
        instruction="Please reason step by step, and put your final answer within \\boxed{{}}.\nThe answer is an integer between 0 and 999 inclusive.",
        default_temperature=0.6,
        default_max_tokens=8000,
        default_epochs=4,
        name="aime_2025_II",
    )
