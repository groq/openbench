from typing import Any, Callable, Optional
from inspect_ai.dataset import Sample, Dataset, hf_dataset, FieldSpec


def record_to_sample(
    max_context_size: Optional[int] = None,
) -> FieldSpec | Callable[[dict[str, Any]], Sample | list[Sample]]:
    """Convert a Needle in a Haystack with Distractors record to an Inspect Sample."""

    def _record_to_sample(record: dict[str, Any]) -> Sample | list[Sample]:
        input_token_count = record.get("token_count")

        # 200 tokens for output
        if (
            max_context_size is not None
            and input_token_count is not None
            and input_token_count + 200 > max_context_size
        ):
            return []

        metadata = {
            "id": record.get("id"),
            "token_count": input_token_count,
            "question": record.get("question"),
        }

        return Sample(
            input=record["input"],
            target=record["needle"],
            metadata=metadata,
        )

    return _record_to_sample


def get_dataset(max_context_size: Optional[int] = None) -> Dataset:
    """Load the Needle in a Haystack with Distractors dataset.

    Args:
        max_context_size: Maximum context size in tokens. Defaults to None.
    Returns:
        Dataset corresponding to the input version.
    """
    return hf_dataset(
        path="kellyhongg/niah-multi-distractors",
        split="train",
        sample_fields=record_to_sample(max_context_size),
    )
