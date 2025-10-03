from inspect_ai.dataset import Dataset, Sample, hf_dataset
from openbench.utils.text import MOCK_AIME_PROMPT


def record_to_sample(record: dict):
    """Convert a MockAIME record to an Inspect Sample."""
    task = MOCK_AIME_PROMPT.format(question=record["question"])
    answer = record.get("answer", "")

    return Sample(
        input=task,
        target=answer,
    )


def get_otis_mock_aime_dataset() -> Dataset:
    """Load the MockAIME dataset."""
    return hf_dataset(
        path="lvogel123/otis-mock-aime-2024-2025",
        split="train",
        sample_fields=record_to_sample,
    )
