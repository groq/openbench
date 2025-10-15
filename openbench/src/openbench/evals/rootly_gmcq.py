"""
GitHub Multiple Choice Questions
Authored by:
Rootly AI Labs
Based on: https://huggingface.co/datasets/TheFloatingString/gmcq

# run code generation
bench eval gmcq --model "groq/llama-3.1-8b-instant" --T subtask=mastodon

If subtask is None, then the entire dataset is used.

Please refer to https://huggingface.co/datasets/TheFloatingString/gmcq for the subtask to use.
There are 6 subtasks as of Tuesday, August 19, 2025, and the None option for the entire dataset:

- bluesky
- chroma
- cloudflare
- duckdb
- mastodon
- tailscale
- None
"""

from typing import Optional
from inspect_ai import Task, task
from inspect_ai.model import (
    GenerateConfig,
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessage,
)
from openbench.utils.mcq import MCQEval, MCQSample


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a GMCQ record to an openbench MCQSample.

    Handles chat-style inputs (list of role/content dicts) and plain strings.
    """
    raw_input = record.get("input", "")
    if isinstance(raw_input, list):
        messages: list[ChatMessage] = []
        for msg in raw_input:
            role = (msg.get("role") or "").lower()
            content = msg.get("content", "")
            if role == "system":
                messages.append(ChatMessageSystem(content=content))
            else:
                # default to user for non-system roles
                messages.append(ChatMessageUser(content=content))
        input_value = messages
    else:
        # treat as plain text question (wrap as a user message for consistency)
        input_value = [ChatMessageUser(content=str(raw_input))]

    target_raw = str(record.get("ideal", "")).strip()
    target = target_raw[0].upper() if target_raw else "A"

    return MCQSample(
        input=input_value,
        target=target,
        metadata={
            "repository_name": record.get("repository_name"),
        },
    )


@task
def rootly_gmcq(subtask: Optional[str] = None) -> Task:  # type: ignore
    """GitHub MCQ (Rootly) using MCQ abstraction with optional subtask filter."""
    dataset_kwargs = {
        "revision": "51c9eace06dd5791e72717bf6ba0348d23857c50",
    }

    # Filter by repository_name via mapper closure if subtask provided
    def mapper_with_filter(record: dict) -> MCQSample | list[MCQSample]:
        repo = record.get("repository_name")
        if subtask is None or (isinstance(subtask, str) and repo in subtask.split(",")):
            return record_to_mcq_sample(record)
        else:
            return []

    return MCQEval(
        name="rootly_gmcq",
        dataset_path="TheFloatingString/gmcq",
        record_to_mcq_sample=mapper_with_filter,
        split="test",
        dataset_kwargs=dataset_kwargs,
        config=GenerateConfig(),
    )
