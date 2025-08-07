"""HealthBench dataset loader."""

import json
from typing import Any, Dict, Optional

import httpx
from inspect_ai.dataset import Dataset, MemoryDataset, Sample


INPUT_PATH = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
INPUT_PATH_HARD = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
INPUT_PATH_CONSENSUS = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"


def record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert a HealthBench record to an Inspect Sample."""
    return Sample(
        id=record.get("prompt_id", ""),
        input=record["prompt"],  # Keep as message list for chat format
        target="",  # No single target - we grade against rubrics
        metadata={
            "rubrics": record["rubrics"],
            "example_tags": record.get("example_tags", []),
            "prompt_id": record.get("prompt_id", ""),
        },
    )


def get_dataset(subset: Optional[str] = None) -> Dataset:
    """Load the HealthBench dataset.

    Args:
        subset: Which subset to load ("hard", "consensus", or None for main)

    Returns:
        Dataset configured for HealthBench evaluation
    """
    # Select URL based on subset
    if subset == "hard":
        url = INPUT_PATH_HARD
    elif subset == "consensus":
        url = INPUT_PATH_CONSENSUS
    elif subset is None:
        url = INPUT_PATH
    else:
        raise ValueError(f"Invalid subset: {subset}")

    # Download and parse the JSONL file
    response = httpx.get(url)
    response.raise_for_status()

    examples = []
    for line in response.text.strip().split("\n"):
        if line:
            examples.append(json.loads(line))

    # Convert to samples
    samples = [record_to_sample(record) for record in examples]

    dataset_name = f"healthbench_{subset}" if subset else "healthbench"
    return MemoryDataset(samples=samples, name=dataset_name)
