from inspect_ai.dataset import Dataset, csv_dataset, Sample, MemoryDataset


def record_to_sample(record: dict) -> Sample:
    """Convert a SimpleQA CSV record to an Inspect Sample."""
    return Sample(
        input=record["problem"],
        target=record["answer"],
        metadata={"metadata": record.get("metadata", "")},
    )


def get_dataset() -> Dataset:
    """Load the SimpleQA dataset.

    Args:
        num_examples: Number of examples to use (None for all)
        n_repeats: Number of times to repeat the dataset (only valid when num_examples is None)
    """
    # Load the full dataset
    dataset = csv_dataset(
        csv_file="https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
        sample_fields=record_to_sample,
        auto_id=True,
        name="simpleqa",
    )

    # Convert to list of samples
    samples = list(dataset)

    return MemoryDataset(samples=samples, name="simpleqa")
