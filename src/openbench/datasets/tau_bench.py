"""
Dataset helpers for tau-bench domains.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from inspect_ai.dataset import MemoryDataset, Sample


def _serialize_task(task) -> dict:
    """
    Convert a tau2 Task (pydantic model) into a JSON-serializable dict.
    """
    return task.model_dump(mode="json")


def _task_prompt(task) -> str:
    """
    Extract a human-readable prompt from the task's user scenario.
    """
    scenario = getattr(task, "user_scenario", None)
    if scenario is None:
        return f"TauBench task {task.id}"
    instructions = getattr(scenario, "instructions", None)
    if instructions is None:
        return f"TauBench task {task.id}"
    return str(instructions)


def get_tau_bench_dataset(
    domain: str,
    *,
    num_trials: int = 1,
    task_ids: Optional[Iterable[str]] = None,
    num_tasks: Optional[int] = None,
) -> MemoryDataset:
    """
    Load tau2 tasks for a domain and expose them as an Inspect dataset.

    Args:
        domain: tau2 domain name (retail, airline, telecom, etc.).
        num_trials: Number of times to repeat each task.
        task_ids: Optional subset of task ids.
        num_tasks: Optional slice of the task list (after filtering).
    """
    from tau2.run import get_tasks as tau2_get_tasks  # type: ignore

    tasks: List = tau2_get_tasks(
        domain, task_ids=list(task_ids) if task_ids else None, num_tasks=num_tasks
    )  # type: ignore[arg-type]
    samples: list[Sample] = []
    for task in tasks:
        serialized = _serialize_task(task)
        prompt = _task_prompt(task)
        for trial in range(1, num_trials + 1):
            samples.append(
                Sample(
                    id=f"{domain}-{task.id}-trial{trial}",
                    input=prompt,
                    target="tau_bench",
                    metadata={
                        "domain": domain,
                        "tau2_task": serialized,
                        "trial": trial,
                    },
                )
            )
    return MemoryDataset(samples=samples, name=f"tau_bench_{domain}")
