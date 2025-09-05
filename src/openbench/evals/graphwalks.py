# src/openbench/evals/graphwalks.py
from __future__ import annotations
from typing import Optional
from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from openbench.datasets.graphwalks import get_dataset
from openbench.scorers.graphwalks import graphwalks_scorer


@task
def graphwalks(
    split: str = "train",
    max_context_size: Optional[int] = None,
    task_type: str = "both",
) -> Task:
    return Task(
        dataset=get_dataset(
            split=split, task_type=task_type, max_context_size=max_context_size
        ),
        solver=[generate()],
        scorer=graphwalks_scorer(),
        name="graphwalks",
        config=GenerateConfig(temperature=0.0, top_p=1.0, max_tokens=8192),
    )


@task
def graphwalks_bfs(
    split: str = "train", max_context_size: Optional[int] = None
) -> Task:
    return Task(
        dataset=get_dataset(
            split=split, task_type="bfs", max_context_size=max_context_size
        ),
        solver=[generate()],
        scorer=graphwalks_scorer(),
        name="graphwalks_bfs",
        config=GenerateConfig(temperature=0.0, top_p=1.0, max_tokens=8192),
    )


@task
def graphwalks_parents(
    split: str = "train", max_context_size: Optional[int] = None
) -> Task:
    return Task(
        dataset=get_dataset(
            split=split, task_type="parents", max_context_size=max_context_size
        ),
        solver=[generate()],
        scorer=graphwalks_scorer(),
        name="graphwalks_parents",
        config=GenerateConfig(temperature=0.0, top_p=1.0, max_tokens=8192),
    )
