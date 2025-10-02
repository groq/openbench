from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.tumlu import get_dataset
from openbench.scorers.tumlu import tumlu_simple_eval_scorer


@task
def tumlu(language: str = "azerbaijani") -> Task:
    return Task(
        dataset=get_dataset(language=language),
        solver=[generate()],
        scorer=tumlu_simple_eval_scorer(),
        name="tumlu",
        config=GenerateConfig(
            temperature=0.5,
        ),
    )
