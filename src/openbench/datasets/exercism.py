"""Exercism subset dataset loader."""

from typing import Any, Dict, List, Optional
from inspect_ai.dataset import Dataset, hf_dataset, Sample, MemoryDataset
import json


def record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert Exercism HF record to an Inspect Sample."""
    language = record.get("language")
    task_name = record.get("task_name")
    record_id = record.get("id") or f"{language}/{task_name}"
    prompt = record.get("prompt", "")

    # Rehydrate task docs
    docs_json = record.get("task_docs_json") or "{}"
    try:
        task_docs = json.loads(docs_json)
        if not isinstance(task_docs, dict):
            task_docs = {}
    except Exception:
        task_docs = {}

    metadata = {
        "language": language,
        "task_name": task_name,
        "test_command": record.get("test_command"),
        "setup_commands": record.get("setup_commands", []),
        "task_docs": task_docs,
        "repo_path": record.get("repo_path"),
    }

    return Sample(id=str(record_id), input=prompt, metadata=metadata)


def get_exercism_dataset(
    languages: Optional[List[str]] = None, split: str = "train"
) -> Dataset:
    """Load the subset of Exercism tasks.

    Args:
        languages: Optional list of languages to include (filters).
        split: Hugging Face split to load (default: "train").

    Returns:
        MemoryDataset with filtered samples.
    """
    dataset = hf_dataset(
        path="lvogel123/exercism",
        split=split,
        sample_fields=record_to_sample,
    )

    samples = list(dataset)

    # Apply language filter
    if languages is not None:
        language_set = set(languages)
        samples = [
            s for s in samples if (s.metadata or {}).get("language") in language_set
        ]

    # Create dataset name based on languages
    if languages:
        language_suffix = "_".join(sorted(languages))
        dataset_name = f"exercism_{language_suffix}"
    else:
        dataset_name = "exercism"

    return MemoryDataset(samples=samples, name=dataset_name)
