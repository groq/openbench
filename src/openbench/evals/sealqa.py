from datetime import date, datetime

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.sealqa import get_dataset
from openbench.scorers.sealqa import sealqa_scorer


@task
def sealqa(
    last_updated: str | date | datetime | None = None,
    grader_model: str = "openai/gpt-4o-mini-2024-07-18",
    subset: str = "seal_hard",
) -> Task:
    """SealQA: MRaising the Bar for Reasoning in Search-Augmented Language Models.

    Based on the paper by Pham et al. (2024).
    Uses model-based grading to assess factual accuracy of responses.

    Args:
        last_updated: Commit SHA, tag, or ISO date alias (e.g. "2025-10-29") in
            `vtllms/sealqa`. Use "latest" for the current HEAD snapshot.
        grader_model: Model to use for grading responses (defaults to openai/gpt-4o-mini-2024-07-18)
        subset: Which SealQA subset to load. Options: "seal_hard", "seal_0", "longseal".

    Returns:
        Task configured for SealQA evaluation
    """
    if last_updated is None:
        raise ValueError(
            "`last_updated` is required because SealQA datasets are versioned by"
            " update date. Use 'latest' or an update date such as '2025-10-29'."
        )

    if isinstance(last_updated, (date, datetime)):
        last_updated_str = last_updated.isoformat()
    else:
        last_updated_str = str(last_updated)

    normalized_last_updated = (
        last_updated_str.lower().replace(" ", "-").replace("/", "-")
    )

    return Task(
        dataset=get_dataset(subset=subset, last_updated=last_updated_str),
        solver=[generate()],
        scorer=sealqa_scorer(model=grader_model),
        name=(
            f"sealqa_{subset}_{normalized_last_updated}"
            if normalized_last_updated != "latest"
            else f"sealqa_{subset}_latest"
        ),
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation for factual QA
        ),
    )
