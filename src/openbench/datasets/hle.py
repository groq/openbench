from inspect_ai.dataset import Dataset, Sample, MemoryDataset, hf_dataset


def record_to_sample(record: dict) -> Sample:
    """Convert an HLE record to an Inspect Sample."""
    # Format the input with the system prompt used in HLE
    input_text = record["question"]

    # Include metadata for tracking
    metadata = {
        "question_id": record["id"],
    }

    # Add image if present (for multi-modal questions)
    if record.get("image"):
        metadata["image_url"] = record["image"]

    return Sample(
        input=input_text,
        target=record["answer"],
        id=record["id"],
        metadata=metadata,
    )


def get_dataset(text_only: bool = False) -> Dataset:
    """Load the HLE (Humanity's Last Exam) dataset.

    Args:
        text_only: If True, filter out multi-modal questions with images

    Returns:
        Dataset with HLE questions and answers
    """
    # Load the dataset from HuggingFace (no 'name' parameter - uses default config)
    dataset = hf_dataset(
        "cais/hle",
        split="test",
        sample_fields=record_to_sample,
    )

    # Convert to list for MemoryDataset
    samples = list(dataset)

    # Filter out image questions if text_only is True
    if text_only:
        samples = [
            s for s in samples if not (s.metadata and s.metadata.get("image_url"))
        ]
        dataset_name = "hle_text"
    else:
        dataset_name = "hle"

    return MemoryDataset(samples=samples, name=dataset_name)
