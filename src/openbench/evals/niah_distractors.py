from typing import Optional
from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.niah_distractors import get_dataset
from openbench.scorers.niah_distractors import niah_distractors_scorer


@task
def niah_distractors(
    max_context_size: Optional[int] = None,
    grader_model: str = "openai/gpt-4.1-2025-04-14",
) -> Task:
    """Needle in a Haystack with Multiple Distractors

    This dataset extends the original Needle in a Haystack (Kamradt, G. (2023)) benchmark by introducing 4 distractors alongside the original setup.
    The haystack passages are drawn from Paul Graham essays.
    Question, needle, and distractors are manually written.

    This is one of the NIAH extensions used in Chroma's Context Rot paper, this is the most challenging variation of NIAH.

    Args:
        max_context_size: Maximum context size in tokens. Defaults to None.
        grader_model: Model to use for grading responses (defaults to gpt-4.1-2025-04-14)

    Returns:
        Task configured for Needle in a Haystack with Multiple Distractors evaluation
    """
    return Task(
        dataset=get_dataset(max_context_size=max_context_size),
        solver=[generate()],
        scorer=niah_distractors_scorer(model=grader_model),
        name="niah_distractors",
        config=GenerateConfig(
            temperature=0.0,
        ),
    )
