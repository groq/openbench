"""
DeepResearch Bench evaluation task for openbench.

Adapted from the original DeepResearch Bench: https://github.com/Ayanami0730/deep_research_bench

This evaluation tests LLMs on research article generation with two phases:
1. RACE: Article quality evaluation (comprehensiveness, insight, instruction-following, readability)
2. FACT: Citation accuracy evaluation

The evaluation follows the original DeepResearch Bench workflow but adapts it to the openbench framework.
"""

import os
from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig

from openbench.datasets.deep_research_bench import get_query
from openbench.solvers.deep_research_bench.deep_research_bench import (
    deep_research_solver,
)
from openbench.scorers.deep_research_bench import deep_research_scorer


@task
def deep_research_bench() -> Task:
    """
    DeepResearch Bench evaluation task.

    Tests LLM research capabilities by having models generate research articles
    in response to complex queries, then evaluating both the quality of the content
    (RACE evaluation) and the accuracy of citations (FACT evaluation).
    """
    # Validate keys immediately when task is created
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError(
            "DeepResearch Bench requires GEMINI_API_KEY environment variable to be set for the RACE and FACT LLM judges. Please set GEMINI_API_KEY and try again."
        )
    if not os.environ.get("JINA_API_KEY"):
        raise ValueError(
            "DeepResearch Bench requires JINA_API_KEY environment variable to be set for the FACT evaluator. Please set JINA_API_KEY and try again."
        )

    return Task(
        dataset=get_query(),
        solver=[deep_research_solver()],
        scorer=deep_research_scorer(),
        name="deep_research_bench",
        config=GenerateConfig(
            max_tokens=36000,  # As specified in the DeepResearch Bench research paper
        ),
    )
