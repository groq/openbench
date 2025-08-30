from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.mmmu_pro import get_dataset
from openbench.scorers.mmlu_pro import mmlu_pro_eval_scorer


@task
def mmmu_pro() -> Task:
    """
    MMMU Pro

    MMMU-Pro is an enhanced multimodal benchmark designed to rigorously assess the true understanding
    capabilities of advanced AI models across multiple modalities. It builds upon the original MMMU
    benchmark by introducing several key improvements that make it more challenging and realistic,
    ensuring that models are evaluated on their genuine ability to integrate and comprehend both visual
    and textual information.

    """
    return Task(
        dataset=get_dataset(subset="standard (10 options)"),
        solver=[generate()],
        scorer=mmlu_pro_eval_scorer(),
        name="mmmu_pro",
        config=GenerateConfig(
            max_tokens=1024,
        ),
    )


@task
def mmmu_pro_vision() -> Task:
    """
    MMMU-Pro (Vision subset)
    """
    return Task(
        dataset=get_dataset(subset="vision"),
        solver=[generate()],
        scorer=mmlu_pro_eval_scorer(),
        name="mmmu_pro_vision",
        config=GenerateConfig(
            max_tokens=1024,
        ),
    )
