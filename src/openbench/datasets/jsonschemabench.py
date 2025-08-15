from datasets import load_dataset  # type: ignore[import-untyped]
from inspect_ai.dataset import Dataset, Sample, MemoryDataset

# Based on JSONSchemaBench paper methodology
JSONSCHEMABENCH_INSTRUCTION = "Generate a valid JSON object that matches the schema below."


def record_to_sample(record: dict) -> Sample:
    """Convert a JSONSchemaBench record to an Inspect Sample."""
    schema = record["json_schema"]
    unique_id = record["unique_id"]
    
    prompt = f"{JSONSCHEMABENCH_INSTRUCTION}\n\nSchema:\n{schema}"
    
    return Sample(
        input=prompt,
        target="",
        metadata={
            "schema": schema,
            "unique_id": unique_id,
        },
    )


def get_dataset(subset: str | None = None, split: str = "test") -> Dataset:
    """Load the JSONSchemaBench dataset from HuggingFace."""
    config = subset if subset else "default"
    
    # Handle paper-style "all splits combined" with HuggingFace syntax
    if split == "all":
        split = "train[:]+val[:]+test[:]"
    
    dataset = load_dataset("epfl-dlab/JSONSchemaBench", config, split=split)
    samples = [record_to_sample(record) for record in dataset]
    
    name = f"jsonschemabench_{config}" if config != "default" else "jsonschemabench"
    if split not in ["test", "train[:]+val[:]+test[:]"]:
        name += f"_{split}"
    
    return MemoryDataset(samples=samples, name=name)