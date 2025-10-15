"""BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents.

Authors: Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han,
Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, Mia Glaese

https://openai.com/index/browsecomp/
"""

from inspect_ai import task, Task
from inspect_ai.solver import generate
from openbench.datasets.browsecomp import get_dataset
from openbench.scorers.browsecomp import browsecomp_scorer


@task
def browsecomp(grader_model: str = "openai/gpt-4.1-2025-04-14") -> Task:
    """BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents.

    This benchmark evaluates model performance on browsing-related tasks
    that require understanding and reasoning about web content.

    Args:
        grader_model: Model to use for grading responses (defaults to gpt-4.1-2025-04-14)

    Returns:
        Task configured for BrowseComp evaluation
    """
    return Task(
        dataset=get_dataset(),
        solver=[generate()],
        scorer=browsecomp_scorer(model=grader_model),
        name="browsecomp",
    )
