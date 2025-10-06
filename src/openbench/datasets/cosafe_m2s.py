"""CoSafe single-turn (m2s) dataset loader."""

from __future__ import annotations

from typing import Optional

from inspect_ai.dataset import Dataset, MemoryDataset, Sample, hf_dataset


AVAILABLE_SUBSETS = ["hyphenize", "numberize", "pythonize"]


def record_to_sample(record: dict) -> Sample:
    prompt = record.get("prompt", "")
    metadata = {
        "prompt": prompt,
        "objective": record.get("objective", ""),
        "id": record.get("id", ""),
    }

    return Sample(
        input=prompt,
        metadata=metadata,
    )


def get_cosafe_m2s_dataset(subset: Optional[str] = None) -> Dataset:
    """Load the CoSafe m2s dataset."""

    if subset:
        dataset = hf_dataset(
            path="lvogel123/m2s-cosafe",
            split=subset,
            sample_fields=record_to_sample,
        )
        samples = list(dataset)
        dataset_name = f"cosafe_m2s_{subset}"
    else:
        all_samples = []
        for name in AVAILABLE_SUBSETS:
            dataset = hf_dataset(
                path="lvogel123/m2s-cosafe",
                split=name,
                sample_fields=record_to_sample,
            )
            all_samples.extend(list(dataset))
        samples = all_samples
        dataset_name = "cosafe_m2s"

    return MemoryDataset(samples=samples, name=dataset_name)
