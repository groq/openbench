from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.simpleqa_verified import get_dataset
from openbench.scorers.simpleqa_verified import simpleqa_verified_scorer


@task
def simpleqa_verified(grader_model: str = "openai/gpt-4.1-2025-04-14") -> Task:
    """SimpleQA Verified: Measuring short-form factuality with improved data quality.

    Curated version from Google DeepMind addressing limitations in the original SimpleQA.
    Uses model-based grading to assess factual accuracy of responses.

    Args:
        grader_model: Model to use for grading responses (defaults to gpt-4.1-2025-04-14)

    Returns:
        Task configured for SimpleQA Verified evaluation
    """
    return Task(
        dataset=get_dataset(),
        solver=[generate()],
        scorer=simpleqa_verified_scorer(model=grader_model),
        name="simpleqa_verified",
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation for factual QA
        ),
    )
