"""
MultiChallenge dataset adapter for OpenBench (Inspect).

Expects a JSONL where each line has:
- QUESTION_ID (str)
- AXIS (str) – e.g., INFERENCE_MEMORY, SELF_COHERENCE, INSTRUCTION_RETENTION
- CONVERSATION (list[{"role": "user"|"assistant", "content": str}])
- TARGET_QUESTION (str)
- PASS_CRITERIA (str) – "YES" or "NO"

Usage in an eval task:
    from openbench.datasets.multichallenge import get_dataset
    ds = get_dataset(jsonl_path="data/benchmark_questions.jsonl", axes=["INFERENCE_MEMORY"])
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from inspect_ai.dataset import FieldSpec, Sample, json_dataset, Dataset
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant


def record_to_sample(
    max_turns: Optional[int] = None,
) -> FieldSpec | Callable[[Dict[str, Any]], Sample]:
    """
    Return a mapping function that converts a MultiChallenge JSONL record
    into an Inspect `Sample`.

    Parameters
    ----------
    max_turns : Optional[int]
        If provided, truncate the conversation to the last `max_turns` messages
        (useful for quick local runs).

    Returns
    -------
    FieldSpec | Callable[[dict], Sample]
        Mapper suitable for `json_dataset(..., sample_fields=...)`.
    """

    def _map(record: Dict[str, Any]) -> Sample:
        convo_raw = record["CONVERSATION"]

        # truncate turn list
        if isinstance(max_turns, int) and max_turns > 0:
            convo_raw = convo_raw[-max_turns:]

        # convert to Inspect chat messages
        messages = []

        for msg in convo_raw:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "user":
                messages.append(ChatMessageUser(content=content))
            elif role == "assistant":
                messages.append(ChatMessageAssistant(content=content))
            else:
                # Fallback: treat unknown roles as user prompts
                messages.append(ChatMessageUser(content=content))

        meta = {
            "question_id": record["QUESTION_ID"],
            "axis": record["AXIS"],
            "target_question": record["TARGET_QUESTION"],
            "pass_criteria": record["PASS_CRITERIA"],
        }

        # Target is optional for judge-based scoring; keep pass_criteria for reference
        return Sample(input=messages, target=record["PASS_CRITERIA"], metadata=meta)

    return _map


def get_dataset(
    limit: Optional[int] = None,
    max_turns: Optional[int] = None,
) -> Dataset:
    """
    Load the MultiChallenge dataset as an Inspect/OpenBench Dataset.

    Args
        limit : Optional[int]: limit the number of samples
        max_turns : Optional[int]: truncate each conversation to the last `max_turns` messages

    Returns:
        Configure Dataset ready to be consumed by an OpenBench task.
    """
    ds = json_dataset(
        json_file="data/multichallenge.jsonl",
        sample_fields=record_to_sample(max_turns=max_turns),
        limit=limit,
        name="multichallenge",
    )

    return ds
