from typing import Any, Callable

from inspect_ai.dataset import Sample, Dataset, MemoryDataset, hf_dataset
from openbench.utils.text import str_to_chat_messages


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
        metadata = {
            "random_string_to_prepend": record.get("random_string_to_prepend"),
            "n_needles": record.get("n_needles"),
            "desired_msg_index": record.get("desired_msg_index"),
            "total_messages": record.get("total_messages"),
            "n_chars": record.get("n_chars"),
        }

        return Sample(
            input=str_to_chat_messages(record["prompt"]),
            target=record["answer"],
            metadata=metadata,
        )

    return _record_to_sample


def get_dataset(needles: int = 2) -> Dataset:
    """Load the MRCR dataset, filtered by number of needles.

    Args:
        needles: Number of needles to include (2, 4, or 8). Defaults to 2.

    Returns:
        Dataset filtered to the requested number of needles.
    """

    # Load from HuggingFace Hub
    dataset = hf_dataset(
        path="openai/mrcr",
        split="train",
        sample_fields=record_to_sample(),
    )

    # Filter to requested needles and return a named in-memory dataset
    samples = [
        s for s in dataset if s.metadata and s.metadata.get("n_needles") == needles
    ]

    return MemoryDataset(samples=samples, name=f"mrcr_{needles}")
