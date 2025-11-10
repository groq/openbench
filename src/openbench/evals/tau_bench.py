"""
Tau-bench eval registrations.
"""

from __future__ import annotations

from typing import Iterable, Optional

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from openbench.datasets.tau_bench import get_tau_bench_dataset
from openbench.scorers.tau_bench import tau_bench_scorer
from openbench.solvers.tau_bench import tau_bench_solver


def _build_tau_bench_task(
    domain: str,
    *,
    user_model: str,
    num_trials: int,
    max_steps: int,
    max_errors: int,
    task_ids: Optional[Iterable[str]] = None,
    num_tasks: Optional[int] = None,
) -> Task:
    dataset = get_tau_bench_dataset(
        domain,
        num_trials=num_trials,
        task_ids=list(task_ids) if task_ids else None,
        num_tasks=num_tasks,
    )
    solver_fn = tau_bench_solver(
        user_model=user_model,
        max_steps=max_steps,
        max_errors=max_errors,
    )
    return Task(
        dataset=dataset,
        solver=[solver_fn],
        scorer=tau_bench_scorer(),
        name=f"tau_bench_{domain}",
        config=GenerateConfig(
            temperature=0.0,
        ),
    )


@task
def tau_bench_retail(
    user_model: str = "openai/gpt-4.1-mini",
    num_trials: int = 1,
    max_steps: int = 200,
    max_errors: int = 10,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
) -> Task:
    """
    Run tau-bench retail tasks with a simulated user and real tool calls.
    """
    return _build_tau_bench_task(
        "retail",
        user_model=user_model,
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        task_ids=task_ids,
        num_tasks=num_tasks,
    )


@task
def tau_bench_airline(
    user_model: str = "openai/gpt-4.1-mini",
    num_trials: int = 1,
    max_steps: int = 200,
    max_errors: int = 10,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
) -> Task:
    return _build_tau_bench_task(
        "airline",
        user_model=user_model,
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        task_ids=task_ids,
        num_tasks=num_tasks,
    )


@task
def tau_bench_telecom(
    user_model: str = "openai/gpt-4.1-mini",
    num_trials: int = 1,
    max_steps: int = 200,
    max_errors: int = 10,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
) -> Task:
    return _build_tau_bench_task(
        "telecom",
        user_model=user_model,
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        task_ids=task_ids,
        num_tasks=num_tasks,
    )
