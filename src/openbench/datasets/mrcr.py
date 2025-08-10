from typing import Any, Callable

from inspect_ai.dataset import Sample, Dataset, MemoryDataset, hf_dataset
from openbench.utils.text import get_chatml_tok_cnt, str_to_chat_messages


def record_to_sample() -> Callable[[dict[str, Any]], Sample]:
    """Create a mapper from MRCR records to Inspect Samples.

    Expected fields in the source record:
    - prompt (str): input to the model
    - answer (str): expected output
    - random_string_to_prepend (str)
    - n_needles (int)
    - desired_msg_index (int)
    - total_messages (int)
    - n_chars (int)
    """

    def _record_to_sample(record: dict[str, Any]) -> Sample:
        chat_messages = str_to_chat_messages(record["prompt"])
        input_tok_cnt = get_chatml_tok_cnt(record["prompt"])
        metadata = {
            "random_string_to_prepend": record.get("random_string_to_prepend"),
            "n_needles": record.get("n_needles"),
            "desired_msg_index": record.get("desired_msg_index"),
            "total_messages": record.get("total_messages"),
            "n_chars": record.get("n_chars"),
            "raw_input_tok_cnt": input_tok_cnt,
        }

        return Sample(
            input=chat_messages,
            target=record["answer"],
            metadata=metadata,
        )

    return _record_to_sample


def get_dataset(needles: int = None, max_context_size: int = None) -> Dataset:
    """Load the MRCR dataset.

    Args:
        needles: Number of needles to include (2, 4, or 8). Defaults to None.
        max_context_size: Maximum context size in tokens. Defaults to None.
    Returns:
        Dataset filtered to the requested number of needles.
    """

    def context_size_filter(x: dict[str, Any]) -> bool:
        return x["raw_input_tok_cnt"] <= max_context_size

    if needles in (2, 4, 8):
        dataset = hf_dataset(
            path="openai/mrcr",
            split="train",
            sample_fields=record_to_sample(),
            data_files=f"{needles}needle.parquet",
        )
    else:
        dataset = hf_dataset(
            path="openai/mrcr",
            split="train",
            sample_fields=record_to_sample(),
        )

    dataset = list(dataset)
    if max_context_size:
        dataset = [s for s in dataset if context_size_filter(s.metadata)]
    name = f"mrcr_{needles}n" if needles else "mrcr_full"
    print(f"Loaded {len(dataset)} samples from {name}")
    return MemoryDataset(samples=dataset, name=name)
