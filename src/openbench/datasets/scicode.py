from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import Sample


def record_to_sample(record):
    return Sample(
        input="problem_id",
        target=record["problem_id"],
        id=record["problem_id"],
        metadata={k: v for k, v in record.items()},
    )


def return_hf_dataset(split: str = "test"):
    dataset = hf_dataset(
        "SciCode1/SciCode",
        split=split,
        sample_fields=record_to_sample,
    )
    return dataset
