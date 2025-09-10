from inspect_ai.dataset import Dataset, hf_dataset, Sample


def record_to_sample(record: dict[str, str]) -> Sample:
    return Sample(
        input=record["question"], target=record["answer"].split("####")[-1].strip()
    )


def get_dataset(split: str = "train", shuffle: bool = False) -> Dataset:
    """
    Load the GSM8K dataset for evaluation.
    Args:
        split: Which dataset split to use - "train" 7,473 questions,
               and "test" contains 1,319 questions.
    Returns:
        Dataset: GSM8K dataset.
    """
    return hf_dataset(
        path="openai/gsm8k",
        name="main",
        split=split,
        sample_fields=record_to_sample,
        shuffle=shuffle,
    )
