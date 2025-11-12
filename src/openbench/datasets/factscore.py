"""FActScore dataset loader.

This dataset prepares biography prompts for the FActScore benchmark using the
official prompt entity lists. The prompts mirror the setup described in the
paper ("Question: Tell me a bio of <entity>.").
"""

from __future__ import annotations

from inspect_ai.dataset import hf_dataset

from inspect_ai.dataset import Dataset, Sample


def record_to_sample(record: dict) -> Sample:
    input = record.get("prompt", "")
    metadata = record.get("metadata", {})

    return Sample(
        input=input,
        metadata=metadata,
    )


def get_dataset() -> Dataset:
    """Load the FActScore dataset"""

    return hf_dataset(
        path="lvogel123/factscore",
        split="train",
        sample_fields=record_to_sample,
    )
