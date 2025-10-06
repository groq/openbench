from inspect_ai.dataset import Dataset, hf_dataset, Sample, MemoryDataset
from typing import Optional


def record_to_sample(record: dict) -> Sample:
    prompt = record.get("prompt", "")
    metadata = {
        "prompt": record.get("prompt", ""),
        "objective": record.get("objective", ""),
        "id": record.get("id", ""),
    }

    return Sample(
        input=prompt,
        target="",
        metadata=metadata,
    )


def get_mhj_m2s_dataset(subset: Optional[str] = None) -> Dataset:
    """
    Load the MHJ-M2S dataset.

    args:
        subset: Optional[str] = None,
            The subset of the MHJ-M2S dataset to use.
            One of: "hyphenize", "numberize", "pythonize".
            If None, all subsets are used.

    Returns:
        Dataset: Configured MHJ-M2S dataset for evaluation
    """
    if subset:
        dataset = hf_dataset(
            path="lvogel123/m2s-mhj",
            split=subset,
            sample_fields=record_to_sample,
        )
        samples = list(dataset)
        dataset_name = f"mhj_m2s_{subset}"
    else:
        available_subsets = ["hyphenize", "numberize", "pythonize"]
        all_samples = []
        for subset in available_subsets:
            dataset = hf_dataset(
                path="lvogel123/m2s-mhj",
                split=subset,
                sample_fields=record_to_sample,
            )
            subset_samples = list(dataset)
            all_samples.extend(subset_samples)
        samples = all_samples
        dataset_name = "mhj_m2s"

    return MemoryDataset(samples=samples, name=dataset_name)
