from inspect_ai.dataset import Sample
from inspect_ai.dataset import hf_dataset
from typing import Any

SUBTASK = None


def record_to_sample_gmcq(record: dict[str, Any]):
    if SUBTASK is None:
        return Sample(
            input=record["input"],
            target=record["ideal"],
        )
    else:
        if record["repository_name"] in SUBTASK.split(","):
            return Sample(
                input=record["input"],
                target=record["ideal"],
            )
        else:
            return []


def load_dataset(subtask):
    global SUBTASK
    SUBTASK = subtask

    dataset = hf_dataset(
        "TheFloatingString/gmcq",
        split="test",
        sample_fields=record_to_sample_gmcq,
        revision="51c9eace06dd5791e72717bf6ba0348d23857c50",
    )
    return dataset
