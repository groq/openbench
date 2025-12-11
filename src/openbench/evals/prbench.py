"""PRBench evaluation implementation."""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.prbench import get_dataset
from openbench.scorers.prbench import prbench_scorer


@task
def prbench(
    split_name: str = "finance",
    grader_model: str = "openai/gpt-4o-mini",
) -> Task:
    """PRBench: Professional Reasoning Benchmark.

    A large-scale expert-annotated benchmark for high-stakes reasoning in professional
    domains. Evaluates model responses against detailed rubrics created by domain experts.

    PRBench consists of:
    - 1,100 expert-authored conversations across Finance and Legal domains
    - 19,356 expert-curated rubric criteria (10–30 per task)
    - Coverage of 114 countries, 47 U.S. jurisdictions, and 25 total professional topics
    - Finance and Legal domain splits

    Based on: https://github.com/scaleapi/PRBench
    Paper: https://scale.com/research/prbench

    Args:
        split_name: Which split to evaluate ("finance", "legal")
        grader_model: Model to use for grading rubrics

    Returns:
        Task configured for PRBench evaluation
    """
    return Task(
        dataset=get_dataset(split_name=split_name),
        solver=[generate()],
        scorer=prbench_scorer(grader_model=grader_model),
        name=f"prbench_{split_name}",
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation for professional reasoning
            max_tokens=8192,  # Allow longer responses for detailed professional explanations
        ),
    )


@task
def prbench_finance(grader_model: str = "openai/gpt-4o-mini") -> Task:
    """PRBench Finance subset.

    Args:
        grader_model: Model to use for grading rubrics

    Returns:
        Task configured for PRBench Finance evaluation
    """
    return Task(
        dataset=get_dataset(split_name="finance"),
        solver=[generate()],
        scorer=prbench_scorer(grader_model=grader_model),
        name="prbench_finance",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=8192,
        ),
    )


@task
def prbench_legal(grader_model: str = "openai/gpt-4o-mini") -> Task:
    """PRBench Legal subset.

    Args:
        grader_model: Model to use for grading rubrics

    Returns:
        Task configured for PRBench Legal evaluation
    """
    return Task(
        dataset=get_dataset(split_name="legal"),
        solver=[generate()],
        scorer=prbench_scorer(grader_model=grader_model),
        name="prbench_legal",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=8192,
        ),
    )
