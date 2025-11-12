"""
Tau-bench eval registrations.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from openbench.datasets.tau_bench import ensure_tau2_data_dir, get_tau_bench_dataset
from openbench.scorers.tau_bench import tau_bench_scorer

# Ensure tau2 data assets are available (and TAU2_DATA_DIR is set) before any tau2 imports.
ensure_tau2_data_dir()

from openbench.solvers.tau_bench import tau_bench_solver  # noqa: E402


def _build_tau_bench_task(
    domain: str,
    *,
    user_model: str,
    max_steps: int,
) -> Task:
    dataset = get_tau_bench_dataset(
        domain,
    )
    solver_fn = tau_bench_solver(
        user_model=user_model,
        max_steps=max_steps,
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
    user_model: str = "openai/gpt-4.1",
    max_steps: int = 200,
) -> Task:
    """
    Run tau-bench retail tasks with a simulated user and real tool calls.
    """
    return _build_tau_bench_task(
        "retail",
        user_model=user_model,
        max_steps=max_steps,
    )


@task
def tau_bench_airline(
    user_model: str = "openai/gpt-4.1",
    max_steps: int = 200,
) -> Task:
    return _build_tau_bench_task(
        "airline",
        user_model=user_model,
        max_steps=max_steps,
    )


@task
def tau_bench_telecom(
    user_model: str = "openai/gpt-4.1",
    max_steps: int = 200,
) -> Task:
    return _build_tau_bench_task(
        "telecom",
        user_model=user_model,
        max_steps=max_steps,
    )
