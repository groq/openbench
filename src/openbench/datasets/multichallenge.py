from __future__ import annotations
from typing import Any, Callable, Dict, Optional
from inspect_ai.dataset import FieldSpec, Sample, hf_dataset, Dataset
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant


def record_to_sample(
    max_turns: Optional[int] = None,
) -> FieldSpec | Callable[[Dict[str, Any]], Sample]:
    """
    Return a mapping function that converts a MultiChallenge JSONL record
    into an Inspect `Sample`.

    Args:
        max_turns : If provided, truncate the conversation to the last `max_turns`
        messages (for quick local runs).
    """

    def _map(record: Dict[str, Any]) -> Sample:
        convo_raw = record["CONVERSATION"]

        # truncate turn list
        if isinstance(max_turns, int) and max_turns > 0:
            convo_raw = convo_raw[-max_turns:]

        # convert to Inspect chat messages
        messages: list[Any] = []

        for msg in convo_raw:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "user":
                messages.append(ChatMessageUser(content=content))
            elif role == "assistant":
                messages.append(ChatMessageAssistant(content=content))
            else:
                # fallback: treat unknown roles as user prompts
                messages.append(ChatMessageUser(content=content))

        meta = {
            "question_id": record["QUESTION_ID"],
            "axis": record["AXIS"],
            "target_question": record["TARGET_QUESTION"],
            "pass_criteria": record["PASS_CRITERIA"],
        }

        # target is optional for judge-based scoring; keep pass_criteria for reference
        return Sample(input=messages, target=record["PASS_CRITERIA"], metadata=meta)

    return _map


def get_dataset(
    max_turns: Optional[int] = None,
) -> Dataset:
    """
    Load the MultiChallenge dataset as an Inspect/OpenBench Dataset.

    Args:
        max_turns : Optional[int]: truncate each conversation to the last `max_turns` messages

    Returns:
        Configure Dataset ready to be consumed by an OpenBench task.
    """
    dataset = hf_dataset(
        path="nmayorga7/multichallenge",
        split="train",  # multichallenge dataset only has train split
        sample_fields=record_to_sample(max_turns=max_turns),
    )
    return dataset
