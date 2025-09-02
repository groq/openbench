from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.longmemeval import get_dataset
from openbench.scorers.longmemeval import longmemeval_scorer


@task
def longmemeval(grader_model: str = "openai/gpt-4.1-2025-04-14") -> Task:
    """LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory

    Based on the paper by Wu et al. (2024).

    Cleaned version of dataset used in Chroma's Context Rot paper.
    Full version containing inputs ~113k tokens.

    Args:
        grader_model: Model to use for grading responses (defaults to gpt-4.1-2025-04-14)

    Returns:
        Task configured for Full Input LongMemEval evaluation
    """
    return Task(
        dataset=get_dataset(input_version="full"),
        solver=[generate()],
        scorer=longmemeval_scorer(model=grader_model),
        name="longmemeval",
        config=GenerateConfig(
            temperature=0.0,
        ),
    )


@task
def longmemeval_focused(grader_model: str = "openai/gpt-4.1-2025-04-14") -> Task:
    """LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory

    Based on the paper by Wu et al. (2024).

    Cleaned version of dataset used in Chroma's Context Rot paper.
    Focused version containing inputs ~300 tokens.

    Args:
        grader_model: Model to use for grading responses (defaults to gpt-4.1-2025-04-14)

    Returns:
        Task configured for Focused Input LongMemEval evaluation
    """
    return Task(
        dataset=get_dataset(input_version="focused"),
        solver=[generate()],
        scorer=longmemeval_scorer(model=grader_model),
        name="longmemeval_focused",
        config=GenerateConfig(
            temperature=0.0,
        ),
    )
