from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.gpt_oss.aime import get_dataset
from openbench.scorers.gpt_oss.aime import gpt_oss_aime_scorer


@task
def gpt_oss_aime25() -> Task:
    return Task(
        dataset=get_dataset(),
        solver=[generate()],
        scorer=gpt_oss_aime_scorer(),
        config=GenerateConfig(
            max_tokens=131_072,
            temperature=1.0,
        ),
        epochs=8,
    )
