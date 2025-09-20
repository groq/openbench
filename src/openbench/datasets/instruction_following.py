"""Instruction Following dataset loader."""

from datasets import load_dataset  # type: ignore[import-untyped]
from inspect_ai.dataset import Dataset, Sample, MemoryDataset


def record_to_sample(record: dict) -> Sample:
    """Convert an IFEval record to an Inspect Sample."""
    return Sample(
        id=str(record.get("key", "")),
        input=record["prompt"],
        target="",
        metadata={
            "instruction_id_list": record["instruction_id_list"],
            "kwargs": record["kwargs"],
        },
    )


def get_dataset() -> Dataset:
    """Load the IFEval dataset from Hugging Face."""
    hf_ds = load_dataset("google/IFEval", split="train")
    return MemoryDataset(list(map(record_to_sample, hf_ds)))
