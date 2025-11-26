"""IFBench dataset loader."""

from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from inspect_ai.dataset import Dataset, MemoryDataset, Sample


def _clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop null values and normalize numeric types."""
    cleaned: dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}
    for key, value in list(cleaned.items()):
        if isinstance(value, float) and value.is_integer():
            cleaned[key] = int(value)
    return cleaned


def _record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert a Hugging Face record into an Inspect Sample."""
    kwargs_list = record.get("kwargs", [])
    return Sample(
        id=str(record.get("key", "")),
        input=record["prompt"],
        target="",
        metadata={
            "instruction_id_list": record["instruction_id_list"],
            "kwargs": [_clean_kwargs(kwargs) for kwargs in kwargs_list],
        },
    )


def get_dataset(split: str = "train") -> Dataset:
    """Load the IFBench dataset from Hugging Face."""
    hf_ds = load_dataset("allenai/IFBench_test", split=split)
    samples = [_record_to_sample(rec) for rec in hf_ds]
    return MemoryDataset(samples=samples, name="ifbench")
