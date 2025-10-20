from inspect_ai.dataset import Dataset, hf_dataset, Sample


def record_to_sample(record: dict) -> Sample:
    prompt = record.get("prompt", "")
    category = record.get("category", "")
    annotations = record.get("metadata", {})
    metadata = {
        "prompt": record.get("prompt", ""),
        "objective": record.get("objective", ""),
        "id": record.get("id", ""),
        "category": category,
        "annotations": annotations,
    }

    return Sample(
        input=prompt,
        metadata=metadata,
    )


def get_mhj_m2s_dataset() -> Dataset:
    """
    Load the MHJ-M2S dataset.

    Returns:
        Dataset: Configured MHJ-M2S dataset for evaluation
    """
    return hf_dataset(
        path="lvogel123/mhj-m2s",
        split="train",
        sample_fields=record_to_sample,
    )
