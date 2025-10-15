from openbench.evals.matharena.matharena import matharena_task
from inspect_ai import Task, task


# instruction: "Please reason step by step, and put your final answer within \\boxed{{}}."
# default_temperature: 0.6
# default_max_tokens: 16000
# strict_parsing: false
# n_problems: 30
# date: "2025-04-08"
# dataset_path: MathArena/brumo_2025
@task
def brumo_2025() -> Task:
    return matharena_task(
        dataset_path="MathArena/brumo_2025",
        instruction="Please reason step by step, and put your final answer within \\boxed{{}}.",
        default_temperature=0.6,
        default_max_tokens=16000,
        default_epochs=4,
        name="brumo_2025",
    )
