import json
from inspect_ai.dataset import Dataset, Sample, MemoryDataset
from typing import Optional


def record_to_sample(record: dict) -> Sample:
    """Convert a realtimeeval JSON record to an Inspect Sample."""
    return Sample(
        input=record["problem"],
        target=record["answer"],
        metadata={"metadata": record.get("metadata", "")},
    )


def get_dataset(json_file: str = "realtimeeval_questions.json") -> Dataset:
    """Load the realtimeeval dataset from a local JSON file.

    Args:
        json_file: Path to the JSON file containing questions and answers
        
    Returns:
        Dataset with questions and answers loaded from the JSON file
    """
    # Load the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert records to samples
    samples = [record_to_sample(record) for record in data]

    return MemoryDataset(samples=samples, name="realtimeeval")