from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.mmlu_pro import get_dataset
from openbench.scorers.mmlu_pro import mmlu_pro_eval_scorer


@task
def mmlu_pro() -> Task:
    """
    MMLU Pro (TIGER-Lab)
    Link: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
    """
    return Task(
        dataset=get_dataset(),
        solver=[generate()],
        scorer=mmlu_pro_eval_scorer(),
        name="mmlu_pro",
        config=GenerateConfig(
            temperature=0.5,
        ),
    )
