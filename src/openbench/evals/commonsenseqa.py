"""CommonsenseQA benchmark evaluation.

Multiple-choice QA requiring commonsense reasoning about everyday concepts.

Dataset: tau/commonsense_qa
Paper: CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge
https://arxiv.org/abs/1811.00937
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


def record_to_sample(record: dict) -> Sample:
    """Convert a CommonsenseQA record to an Inspect Sample."""
    choices = record["choices"]
    return Sample(
        input=record["question"],
        target=record["answerKey"],
        choices=choices["text"],
        metadata={"question_concept": record.get("question_concept", "")},
    )


@task
def commonsenseqa(split: str = "validation") -> Task:
    """CommonsenseQA: Commonsense reasoning benchmark."""
    return Task(
        dataset=hf_dataset(
            path="tau/commonsense_qa",
            split=split,
            sample_fields=record_to_sample,
            trust=True,
        ),
        solver=[multiple_choice()],
        scorer=choice(),
        config=GenerateConfig(temperature=0.0, max_tokens=1024),
    )
