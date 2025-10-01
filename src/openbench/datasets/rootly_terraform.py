from inspect_ai.dataset import Sample
from inspect_ai.dataset import hf_dataset
from typing import Any

SUBTASK = None


def record_to_sample_terraform(record: dict[str, Any]):
    return Sample(
        input=record["input"],
        target=record["ideal"],
    )


def load_rootly_terraform_dataset(subtask):
    global SUBTASK
    SUBTASK = subtask

    print("subtask", SUBTASK)

    if SUBTASK is None or SUBTASK == "rootly-terraform-azure-k8s-mcq":
        dataset = hf_dataset(
            "TheFloatingString/rootly_terraform_azure_k8s_1",
            split="test",
            sample_fields=record_to_sample_terraform,
            revision="2852e65fd8dc1b5302b83899381b1c086dd119ba",
        )

    elif SUBTASK == "rootly-terraform-s3-security-mcq":
        dataset = hf_dataset(
            "TheFloatingString/s3_tf_s3_security_mcq",
            split="test",
            sample_fields=record_to_sample_terraform,
            revision="a4fc90b54b1f191c1a13224dedddc0c9eb881a2d",
        )
    return dataset
