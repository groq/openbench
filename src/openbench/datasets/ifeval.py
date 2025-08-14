from inspect_ai.dataset import Sample
from typing import Any
from inspect_ai.dataset import hf_dataset


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert IFEval dataset record to inspect Sample."""

    def create_list_of_not_none_dicts(
        dict_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            only_keep_not_none_values_in_dict(d) for d in dict_list if d is not None
        ]

    def only_keep_not_none_values_in_dict(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if v is not None}

    return Sample(
        input=record["prompt"],
        target=record.get("expected_output", ""),
        metadata={
            "key": record.get("key", ""),
            "prompt": record.get("prompt", ""),
            "instruction_id_list": record.get("instruction_id_list", ""),
            "kwargs": create_list_of_not_none_dicts(record.get("kwargs", [])),
        },
    )


def get_ifeval_dataset():
    # IFEval dataset from Hugging Face
    dataset = hf_dataset(
        path="google/IFEval",
        split="train",
        sample_fields=record_to_sample,
    )
    return dataset
