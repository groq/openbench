from __future__ import annotations
from typing import Any, Optional, Callable
from inspect_ai.dataset import Dataset, Sample, hf_dataset, FieldSpec
from openbench.utils.text import get_token_count

_ALLOWED = {"bfs", "parents"}


def record_to_sample(
    allowed: Optional[set[str]] = None,
    max_context_size: Optional[int] = None,
) -> FieldSpec | Callable[[dict[str, Any]], Sample | list[Sample]]:
    """Create a mapper from GraphWalks records to Inspect Samples.

    Expected fields in the source record:
    - prompt (str): input to the model
    - answer_nodes (list[str]): expected output
    - prompt_chars (int): input prompt character count
    - problem_type (str): "parents" or "bfs"
    """

    def _record_to_sample(record: dict[str, Any]) -> Sample | list[Sample]:
        problem_type = (record.get("problem_type") or "").strip().lower()

        # problem type filter
        if allowed is not None and problem_type not in allowed:
            return []

        # calculate tokens
        prompt = str(record.get("prompt"))
        tok_cnt = int(get_token_count(prompt))

        # token filter if max_context_size is provided
        if max_context_size is not None and tok_cnt > max_context_size:
            return []

        metadata = {
            "problem_type": problem_type,
            "n_chars": record.get("prompt_chars"),
            "raw_input_tok_cnt": tok_cnt,
            "target": record.get("answer_nodes"),
        }

        return Sample(
            input=prompt,
            target=record.get("answer_nodes", []),
            metadata=metadata,
        )

    return _record_to_sample


def get_dataset(
    split: str = "train",
    task_type: str = "both",
    max_context_size: Optional[int] = None,
) -> Dataset:
    task = (task_type or "both").strip().lower()
    if task in ("both", "all", "*"):
        allowed = None
    elif task in _ALLOWED:
        allowed = {task}
    else:
        raise ValueError("task_type must be one of 'bfs', 'parents', 'both'")

    return hf_dataset(
        path="openai/graphwalks",
        split=split,
        sample_fields=record_to_sample(
            allowed=allowed, max_context_size=max_context_size
        ),
    )
