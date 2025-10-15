from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.simpleqa import get_dataset
from openbench.scorers.simpleqa import simpleqa_scorer


@task
def simpleqa(grader_model: str = "openai/gpt-4.1-2025-04-14") -> Task:
    """SimpleQA: Measuring short-form factuality in large language models.

    Based on the paper by Wei et al. (2024).
    Uses model-based grading to assess factual accuracy of responses.

    Args:
        grader_model: Model to use for grading responses (defaults to gpt-4.1-2025-04-14)

    Returns:
        Task configured for SimpleQA evaluation
    """
    return Task(
        dataset=get_dataset(),
        solver=[generate()],
        scorer=simpleqa_scorer(model=grader_model),
        name="simpleqa",
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation for factual QA
        ),
    )
