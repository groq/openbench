from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
from openbench.utils.mcq import MCQSample
import warnings

# Dataset configurations
DATASETS = {
    "azure-k8s-mcq": {
        "path": "TheFloatingString/rootly_terraform_azure_k8s_1",
        "revision": "2852e65fd8dc1b5302b83899381b1c086dd119ba",
    },
    "s3-security-mcq": {
        "path": "TheFloatingString/s3_tf_s3_security_mcq",
        "revision": "a4fc90b54b1f191c1a13224dedddc0c9eb881a2d",
    },
}


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a Rootly Terraform record to an OpenBench MCQSample."""
    # Convert message dicts to ChatMessage objects
    messages: list[ChatMessage] = []
    for msg in record["input"]:
        if msg["role"] == "system":
            messages.append(ChatMessageSystem(content=msg["content"]))
        elif msg["role"] == "user":
            messages.append(ChatMessageUser(content=msg["content"]))

    return MCQSample(
        input=messages,
        target=record["ideal"],
    )


def get_dataset_config(subtask: str | None = None) -> dict:
    """Get dataset configuration for a given subtask."""
    if subtask is None:
        warnings.warn(
            "No subtask provided, defaulting to azure-k8s-mcq. "
            "If you want to use a specific subtask, use the `-T subtask=<subtask: str>` flag in the OpenBench CLI.",
            UserWarning,
        )

        subtask = "azure-k8s-mcq"

    if subtask not in DATASETS:
        valid_subtasks = ", ".join(DATASETS.keys())
        raise ValueError(
            f"Unknown subtask: {subtask}. Valid subtasks are: {valid_subtasks}"
        )

    return DATASETS[subtask]
