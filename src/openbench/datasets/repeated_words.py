from typing import Any, Callable, Optional
from inspect_ai.dataset import Sample, Dataset, hf_dataset, FieldSpec
from openbench.utils.text import get_token_count


def record_to_sample(
    max_context_size: Optional[int] = None,
) -> FieldSpec | Callable[[dict[str, Any]], Sample | list[Sample]]:
    """Convert a Repeated Words record to an Inspect Sample."""

    def _record_to_sample(record: dict[str, Any]) -> Sample | list[Sample]:
        input_tok_cnt = get_token_count(record["input"])
        max_output_tokens = input_tok_cnt * 2

        if (
            max_context_size is not None
            and input_tok_cnt + max_output_tokens > max_context_size
        ):
            return []

        metadata = {
            "id": record.get("id"),
            "max_output_tokens": max_output_tokens,
        }

        return Sample(
            input=record["input"],
            target=record["target"],
            metadata=metadata,
        )

    return _record_to_sample


def get_dataset(max_context_size: Optional[int] = None) -> Dataset:
    """Load the Repeated Words dataset.

    Args:
        max_context_size: Maximum context size in tokens. Defaults to None.
    Returns:
        Dataset corresponding to the input version.
    """
    return hf_dataset(
        path="kellyhongg/repeated-words",
        split="train",
        sample_fields=record_to_sample(max_context_size),
    )
