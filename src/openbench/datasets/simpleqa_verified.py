"""
SimpleQA Verified dataset from Kaggle.
Reference: https://www.kaggle.com/datasets/deepmind/simpleqa-verified
"""

import kagglehub  # type: ignore[import-untyped]
from inspect_ai.dataset import Dataset, csv_dataset, MemoryDataset
import os

from openbench.datasets.simpleqa import record_to_sample


def get_dataset() -> Dataset:
    """Load the SimpleQA Verified dataset from Kaggle.
    This downloads the dataset from Kaggle and loads it as a CSV dataset.
    """
    # Download the dataset from Kaggle
    path = kagglehub.dataset_download("deepmind/simpleqa-verified")

    # Find the CSV file in the downloaded path
    csv_file = os.path.join(path, "simpleqa_verified.csv")

    # Load the dataset
    dataset = csv_dataset(
        csv_file=csv_file,
        sample_fields=record_to_sample,
        auto_id=True,
        name="simpleqa_verified",
    )

    # Convert to list of samples
    samples = list(dataset)

    return MemoryDataset(samples=samples, name="simpleqa_verified")
