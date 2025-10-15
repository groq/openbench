from openbench.evals.matharena.matharena import matharena_task
from inspect_ai import Task, task


# instruction: "Please reason step by step, and put your final answer within \\boxed{{}}.\nThe answer is an integer between 0 and 999 inclusive."
# default_temperature: 0.6
# default_max_tokens: 8000
# strict_parsing: false
# n_problems: 15
# date: "2024-02-07"
# dataset_path: MathArena/aime_2024_I
@task
def aime_2024_I() -> Task:
    return matharena_task(
        dataset_path="MathArena/aime_2024_I",
        instruction="Please reason step by step, and put your final answer within \\boxed{{}}.\nThe answer is an integer between 0 and 999 inclusive.",
        default_temperature=0.6,
        default_max_tokens=8000,
        default_epochs=4,
        name="aime_2024_I",
    )
