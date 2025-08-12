from inspect_ai.dataset import Dataset, csv_dataset, Sample, MemoryDataset


def record_to_sample(record: dict) -> Sample:
    """Convert a MATH CSV record to an Inspect Sample."""
    return Sample(
        input=record["Question"],
        target=record["Answer"],
        metadata={},
    )


def get_dataset(split: str = "math_test") -> Dataset:
    """Load the MATH dataset.

    Args:
        split: Which dataset split to use - "math_test" for full 5000 problems,
               or "math_500_test" for 500 problem subset
    """
    # Load the dataset from OpenAI's blob storage
    dataset = csv_dataset(
        csv_file=f"https://openaipublic.blob.core.windows.net/simple-evals/{split}.csv",
        sample_fields=record_to_sample,
        auto_id=True,
        name=split,
    )

    # Convert to list of samples
    samples = list(dataset)

    return MemoryDataset(samples=samples, name=split)
