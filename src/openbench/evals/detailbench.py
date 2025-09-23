from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig

from openbench.datasets.detailbench import get_dataset
from openbench.scorers.detailbench import detailbench_scorer


@task
def detailbench(grader_model: str = "openai/gpt-5-mini-2025-08-07") -> Task:
    """
    Tests whether LLMs notify users about wrong facts in a text while they are tasked to translate said text.
    """
    return Task(
        dataset=get_dataset(),
        solver=[generate()],
        scorer=detailbench_scorer(model=grader_model),
        name="detailbench",
        config=GenerateConfig(
            temperature=1.0  # This should be model-specific in practice
        ),
    )
