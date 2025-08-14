# src/openbench/evals/graphwalks.py
from __future__ import annotations

from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.graphwalks import get_dataset
from openbench.scorers.graphwalks import graphwalks_scorer


@task
def graphwalks(split: str = "train") -> Task:
    return Task(
        dataset=get_dataset(split=split, task_type="both"),
        solver=[generate()],
        scorer=graphwalks_scorer(),
        name="graphwalks",
        config=GenerateConfig(temperature=0.0, top_p=1.0, max_tokens=256),
    )


@task
def graphwalks_bfs(split: str = "train") -> Task:
    return Task(
        dataset=get_dataset(split=split, task_type="bfs"),
        solver=[generate()],
        scorer=graphwalks_scorer(),
        name="graphwalks_bfs",
        config=GenerateConfig(temperature=0.0, top_p=1.0, max_tokens=256),
    )


@task
def graphwalks_parents(split: str = "train") -> Task:
    return Task(
        dataset=get_dataset(split=split, task_type="parents"),
        solver=[generate()],
        scorer=graphwalks_scorer(),
        name="graphwalks_parents",
        config=GenerateConfig(temperature=0.0, top_p=1.0, max_tokens=256),
    )
