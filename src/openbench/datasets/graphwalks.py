# src/openbench/datasets/graphwalks.py
from __future__ import annotations
from typing import Any, Optional
from inspect_ai.dataset import Dataset, Sample, hf_dataset

_ALLOWED = {"bfs", "parents"}


def record_to_sample(
    record: dict[str, Any], *, allowed: Optional[set[str]] = None
) -> Optional[Sample]:
    """
    Map one HF row to an Inspect Sample.
    If `allowed` is provided, drop rows whose problem_type isn't in it by returning None.
    """
    problem_type = (record.get("problem_type") or "").strip().lower()

    # Filter here by returning None (row is skipped)
    if allowed is not None and problem_type not in allowed:
        return None

    gold = record.get("answer", record.get("answer_nodes", []))

    return Sample(
        input=record["prompt"],
        target=gold,
        metadata={
            "problem_type": problem_type,
            "prompt_chars": record.get("prompt_chars"),
        },
    )


def get_dataset(split: str = "train", task_type: str = "both") -> Dataset:
    """
    task_type: 'bfs' | 'parents' | 'both' (default: keep all)
    """
    task = (task_type or "both").strip().lower()
    if task in ("both", "all", "*"):
        allowed = None
        name_suffix = "all"
    elif task in _ALLOWED:
        allowed = {task}
        name_suffix = task
    else:
        raise ValueError("task_type must be one of 'bfs', 'parents', 'both'")

    return hf_dataset(
        path="openai/graphwalks",
        split=split,
        sample_fields=lambda rec, _allowed=allowed: record_to_sample(
            rec, allowed=_allowed
        ),
        name=f"graphwalks_{name_suffix}",
    )
