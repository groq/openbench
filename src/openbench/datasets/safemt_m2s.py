"""SafeMT single-turn (m2s) dataset loader."""

from __future__ import annotations

from inspect_ai.dataset import Dataset, Sample, hf_dataset


def record_to_sample(record: dict) -> Sample:
    prompt = record.get("prompt", "")
    category = record.get("category", "")
    annotations = record.get("metadata", {})
    metadata = {
        "prompt": prompt,
        "objective": record.get("objective", ""),
        "id": record.get("id", ""),
        "category": category,
        "annotations": annotations,
    }

    return Sample(
        input=prompt,
        metadata=metadata,
    )


def get_safemt_m2s_dataset() -> Dataset:
    """Load the SafeMT m2s dataset (actor-attack conversion)."""

    return hf_dataset(
        path="lvogel123/safemt-attack-600-m2s",
        split="train",
        sample_fields=record_to_sample,
    )
