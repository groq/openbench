from typing import Any, Callable

from inspect_ai.dataset import Sample, Dataset, hf_dataset, FieldSpec


def record_to_sample(
    input_version: str = "full",
) -> FieldSpec | Callable[[dict[str, Any]], Sample | list[Sample]]:
    """Convert a LongMemEval record to an Inspect Sample.

    Args:
        input_version: input version to use (full ~113k tokens or focused ~300k tokens). Defaults to "full".
    Returns:
        FieldSpec or callable to convert a record to a Sample.
    """

    input_col = f"{input_version}_input"
    token_count_col = f"{input_version}_input_tokens"

    def _record_to_sample(record: dict[str, Any]) -> Sample | list[Sample]:
        metadata = {
            "question": record.get("question"),
            "token_count": record.get(token_count_col),
        }

        return Sample(
            input=record[input_col],
            target=record["answer"],
            metadata=metadata,
        )

    return _record_to_sample


def get_dataset(input_version: str = "full") -> Dataset:
    """Load the LongMemEval dataset.

    Args:
        input_version: input version to use (full ~113k tokens or focused ~300k tokens). Defaults to "full".
    Returns:
        Dataset corresponding to the input version.
    """
    return hf_dataset(
        path="kellyhongg/cleaned-longmemeval-s",
        split="train",
        sample_fields=record_to_sample(input_version),
    )
