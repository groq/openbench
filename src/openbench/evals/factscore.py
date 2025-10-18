"""FActScore benchmark task definition."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.factscore import get_dataset
from openbench.scorers.factscore import factscore_scorer


@task
def factscore(
    split: str = "unlabeled",
    num_examples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    prompt_template: str = "Question: Tell me a bio of {topic}.",
    cache_root: str | None = None,
    model_name: str = "gpt-4o-mini",
    grader_model: str = "gpt-4o-mini",
    knowledge_source: str | None = "enwiki-20230401",
    gamma: int = 10,
    passages: int = 8,
) -> Task:
    """Construct the FActScore evaluation task.

    Args:
        split: ``"labeled"`` (183 topics) or ``"unlabeled"`` (500 topics).
        num_examples: Optional limit on the number of entities to evaluate.
        shuffle: Shuffle entities prior to sampling (use with ``seed``).
        seed: RNG seed used when shuffling.
        prompt_template: Prompt format supplied to the system model.
        cache_root: Directory containing cached FActScore assets. Defaults to
            ``OPENBENCH_FACTSCORE_CACHE`` or ``~/.openbench/factscore``.
        model_name: Model used to generate biographies.
        grader_model: OpenAI model identifier used by FactScoreLite (defaults to gpt-4o-mini).
        knowledge_source: Name or path of the SQLite knowledge base (default
            Wikipedia snapshot).
        gamma: Length penalty parameter from the original paper.
        passages: Number of passages retrieved per topic.

    Returns:
        Fully configured Inspect ``Task``.
    """

    dataset = get_dataset(
        split=split,
        num_examples=num_examples,
        shuffle=shuffle,
        seed=seed,
        prompt_template=prompt_template,
        cache_root=cache_root,
    )

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
