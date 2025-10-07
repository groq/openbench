from inspect_ai.dataset import Dataset, Sample, hf_dataset, MemoryDataset
from typing import Optional
from openbench.utils.text import MOCK_AIME_PROMPT


def record_to_sample(record: dict) -> Sample:
    """Convert a MockAIME record to an Inspect Sample."""
    task = MOCK_AIME_PROMPT.format(question=record["question"])
    answer = record.get("answer", "")
    year = record.get("year", "")
    annotations = record.get("annotations", {})
    mathematical_topic = (
        annotations.get("mathematical_topic", "")
        if isinstance(annotations, dict)
        else ""
    )

    return Sample(
        input=task,
        target=answer,
        metadata={
            "year": year,
            "mathematical_topic": mathematical_topic,
            "annotations": annotations,
        },
    )


def get_otis_mock_aime_dataset(year: Optional[str] = None) -> Dataset:
    """Load the MockAIME dataset."""
    dataset = hf_dataset(
        path="lvogel123/otis-mock-aime-24-25",
        split="train",
        sample_fields=record_to_sample,
    )
    samples = list(dataset)
    if year is not None:
        samples = [
            sample for sample in samples if (sample.metadata or {}).get("year") == year
        ]
    name = f"otis_mock_aime_{year}" if year is not None else "otis_mock_aime"
    return MemoryDataset(samples=samples, name=name)


def get_otis_mock_aime_2024_dataset() -> Dataset:
    """Load the MockAIME dataset filtered for 2024."""
    return get_otis_mock_aime_dataset(year="2024")


def get_otis_mock_aime_2025_dataset() -> Dataset:
    """Load the MockAIME dataset filtered for 2025."""
    return get_otis_mock_aime_dataset(year="2025")
