from inspect_ai import Task, task, Epochs
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import system_message, generate
from openbench.utils.text import SIMPLE_EVALS_SYSTEM_MESSAGE
from openbench.datasets.gpqa import get_dataset
from openbench.scorers.mcq import robust_mcq_scorer


# There is one difference between this and the original gpqa simple eval - the prompts are not reshuffled for every epoch. Shouldn't be that big of a deal, but worth noting.
@task
def gpqa_diamond() -> Task:
    return Task(
        dataset=get_dataset(),
        solver=[system_message(SIMPLE_EVALS_SYSTEM_MESSAGE), generate()],
        scorer=robust_mcq_scorer(),
        name="gpqa_diamond",
        config=GenerateConfig(temperature=0.5),
        epochs=Epochs(10),
    )
