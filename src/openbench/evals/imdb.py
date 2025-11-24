"""IMDB sentiment classification benchmark.

Binary sentiment classification on movie reviews.

Dataset: stanfordnlp/imdb
Paper: Learning Word Vectors for Sentiment Analysis (Maas et al., 2011)
https://aclanthology.org/P11-1015/
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

LABELS = ["negative", "positive"]


def record_to_sample(record: dict) -> Sample:
    """Convert an IMDB record to an Inspect Sample."""
    label_idx = record["label"]
    return Sample(
        input=f"Classify the sentiment of this movie review as positive or negative.\n\nReview: {record['text']}",
        target=chr(ord("A") + label_idx),  # A=negative, B=positive
        choices=LABELS,
    )


@task
def imdb(split: str = "test") -> Task:
    """IMDB: Sentiment classification benchmark."""
    return Task(
        dataset=hf_dataset(
            path="stanfordnlp/imdb",
            split=split,
            sample_fields=record_to_sample,
            trust=True,
        ),
        solver=[multiple_choice()],
        scorer=choice(),
        config=GenerateConfig(temperature=0.0, max_tokens=1024),
    )
