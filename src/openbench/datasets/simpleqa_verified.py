"""
SimpleQA Verified dataset from Hugging Face.
Reference: https://huggingface.co/datasets/google/simpleqa-verified
"""

from openbench.datasets.simpleqa import record_to_sample
from inspect_ai.dataset import Dataset, MemoryDataset, hf_dataset


def get_dataset() -> Dataset:
    """Load the SimpleQA Verified dataset from Hugging Face.
    This downloads the dataset from Hugging Face Hub and loads it.
    """

    # downloading the dataset from Hugging Face Hub
    dataset = hf_dataset(
        path="google/simpleqa-verified",
        name="simpleqa_verified",
        split="eval",
        sample_fields=record_to_sample,
        auto_id=True,
    )

    # Convert to list of samples
    samples = list(dataset)

    return MemoryDataset(samples=samples, name="simpleqa_verified")
