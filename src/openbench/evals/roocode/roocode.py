"""
Roo-Code-Evals evaluation tasks.

This module provides inspect_ai Tasks for evaluating coding abilities across
multiple programming languages using the Roo-Code-Evals benchmark.
"""

from typing import Optional, List
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from openbench.datasets.roocode import (
    get_roocode_dataset,
    get_roocode_python_dataset,
    get_roocode_javascript_dataset,
    get_roocode_go_dataset,
    get_roocode_java_dataset,
    get_roocode_rust_dataset,
)
from openbench.solvers.roocode_agent_solver import roocode_agent_solver
from openbench.scorers.roocode import roocode_scorer


TASK_DIR = Path(__file__).parent
COMPOSE_PATH = (TASK_DIR / "compose.yaml").resolve()


@task
def roocode(
    languages: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Task:
    """
    Roo-Code-Evals: Multi-language coding benchmark.

    Evaluates coding abilities across multiple programming languages using
    real-world coding exercises from the Roo-Code-Evals repository.

    Args:
        languages: List of programming languages to include (python, go, javascript, java, rust).
                  If None, includes all supported languages.
        tasks: List of specific task names to include. If None, includes all tasks.
        limit: Maximum number of samples to include across all languages.

    Returns:
        Task configured for Roo-Code evaluation
    """
    dataset = get_roocode_dataset(
        languages=languages,
        tasks=tasks,
        limit=limit,
    )

    return Task(
        dataset=dataset,
        solver=roocode_agent_solver(),
        scorer=roocode_scorer(),
        sandbox=("docker", str(COMPOSE_PATH)),
        config=GenerateConfig(
            max_tokens=4096,  # Allow longer code responses
        ),
        time_limit=300,  # 5 minute time limit
    )


@task
def roocode_python(
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Task:
    """
    Roo-Code-Evals: Python coding tasks only.

    Args:
        tasks: List of specific task names to include. If None, includes all Python tasks.
        limit: Maximum number of samples to include.

    Returns:
        Task configured for Python-only Roo-Code evaluation
    """
    dataset = get_roocode_python_dataset(
        tasks=tasks,
        limit=limit,
    )

    return Task(
        dataset=dataset,
        solver=roocode_agent_solver(),
        scorer=roocode_scorer(),
        sandbox=("docker", str(COMPOSE_PATH)),
        config=GenerateConfig(
            max_tokens=4096,
            temperature=0.1,
        ),
        time_limit=300,  # 5 minute time limit
    )


@task
def roocode_javascript(
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Task:
    """
    Roo-Code-Evals: JavaScript coding tasks only.

    Args:
        tasks: List of specific task names to include. If None, includes all JavaScript tasks.
        limit: Maximum number of samples to include.

    Returns:
        Task configured for JavaScript-only Roo-Code evaluation
    """
    dataset = get_roocode_javascript_dataset(
        tasks=tasks,
        limit=limit,
    )

    return Task(
        dataset=dataset,
        solver=roocode_agent_solver(),
        scorer=roocode_scorer(),
        sandbox=("docker", str(COMPOSE_PATH)),
        config=GenerateConfig(
            max_tokens=4096,
            temperature=0.1,
        ),
        time_limit=300,  # 5 minute time limit
    )


@task
def roocode_go(
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Task:
    """
    Roo-Code-Evals: Go coding tasks only.

    Args:
        tasks: List of specific task names to include. If None, includes all Go tasks.
        limit: Maximum number of samples to include.

    Returns:
        Task configured for Go-only Roo-Code evaluation
    """
    dataset = get_roocode_go_dataset(
        tasks=tasks,
        limit=limit,
    )

    return Task(
        dataset=dataset,
        solver=roocode_agent_solver(),
        scorer=roocode_scorer(),
        sandbox=("docker", str(COMPOSE_PATH)),
        config=GenerateConfig(
            max_tokens=4096,
            temperature=0.1,
        ),
        time_limit=300,  # 5 minute time limit
    )


@task
def roocode_java(
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Task:
    """
    Roo-Code-Evals: Java coding tasks only.

    Args:
        tasks: List of specific task names to include. If None, includes all Java tasks.
        limit: Maximum number of samples to include.

    Returns:
        Task configured for Java-only Roo-Code evaluation
    """
    dataset = get_roocode_java_dataset(
        tasks=tasks,
        limit=limit,
    )

    return Task(
        dataset=dataset,
        solver=roocode_agent_solver(),
        scorer=roocode_scorer(),
        sandbox=("docker", str(COMPOSE_PATH)),
        config=GenerateConfig(
            max_tokens=4096,
            temperature=0.1,
        ),
        time_limit=300,  # 5 minute time limit
    )


@task
def roocode_rust(
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Task:
    """
    Roo-Code-Evals: Rust coding tasks only.

    Args:
        tasks: List of specific task names to include. If None, includes all Rust tasks.
        limit: Maximum number of samples to include.

    Returns:
        Task configured for Rust-only Roo-Code evaluation
    """
    dataset = get_roocode_rust_dataset(
        tasks=tasks,
        limit=limit,
    )

    return Task(
        dataset=dataset,
        solver=roocode_agent_solver(),
        scorer=roocode_scorer(),
        sandbox=("docker", str(COMPOSE_PATH)),
        config=GenerateConfig(
            max_tokens=4096,
            temperature=0.1,
        ),
        time_limit=300,  # 5 minute time limit
    )
