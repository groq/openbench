"""FActScore benchmark task definition."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.factscore import get_dataset
from openbench.scorers.factscore import factscore_scorer, is_factscore_available


@task
def factscore(
    cache_root: str | None = None,
    grader_model: str = "gpt-4o-mini",
    knowledge_source: str | None = "enwiki-20230401",
    gamma: int = 10,
    passages: int = 8,
) -> Task:
    """Construct the FActScore evaluation task.

    Args:
        cache_root: Directory containing cached FActScore assets. Defaults to
            ``OPENBENCH_FACTSCORE_CACHE`` or ``~/.openbench/factscore``.
        grader_model: OpenAI model identifier used by FactScoreLite (defaults to gpt-4o-mini).
        knowledge_source: Name or path of the SQLite knowledge base (default
            Wikipedia snapshot).
        gamma: Length penalty parameter from the original paper.
        passages: Number of passages retrieved per topic.

    Returns:
        Fully configured Inspect ``Task``.
    """

    if not is_factscore_available():
        raise RuntimeError(
            "FactScoreLite package is required to run the factscore benchmark."
        )

    dataset = get_dataset()

    scorer = factscore_scorer(
        model_name=grader_model,
        knowledge_source=knowledge_source,
        gamma=gamma,
        cache_root=cache_root,
        passages=passages,
    )

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=scorer,
        name="factscore",
        config=GenerateConfig(
            temperature=0.2,
            max_tokens=8192,
        ),
    )
