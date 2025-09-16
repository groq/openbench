"""
Roo-Code-Evals evaluation tasks.

This module provides inspect_ai Tasks for evaluating coding abilities across
multiple programming languages using the Roo-Code-Evals benchmark.
"""

from typing import Optional, List
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from openbench.datasets.exercism import get_exercism_dataset
from openbench.solvers.exercism_solver import exercism_solver
from openbench.scorers.exercism import exercism_scorer


TASK_DIR = Path(__file__).parent
COMPOSE_PATH = (TASK_DIR / "compose.yaml").resolve()


@task
def exercism(
    languages: Optional[List[str]] = None,
    harness: str = "opencode",
) -> Task:
    """
    Exercism: Multi-language coding benchmark.

    Evaluates coding abilities across multiple programming languages using
    real-world coding exercises from the Exercism Tasks.

    Args:
        languages: List of programming languages to include (python, go, javascript, java, rust).
                  If None, includes all supported languages.
        harness: CLI harness to use for code evaluation ('aider', 'opencode', 'claude', 'roo').
                Defaults to 'opencode'. Can also be set via --harness flag.

    Returns:
        Task configured for Exercism evaluation
    """
    dataset = get_exercism_dataset(languages=languages)

    # Add harness to each sample's metadata so the solver can access it
    for sample in dataset:
        if not hasattr(sample, "metadata") or sample.metadata is None:
            sample.metadata = {}
        sample.metadata["harness"] = harness

    return Task(
        dataset=dataset,
        solver=exercism_solver(),
        scorer=exercism_scorer(),
        sandbox=("docker", str(COMPOSE_PATH)),
        config=GenerateConfig(
            max_tokens=4096,  # Allow longer code responses
        ),
        time_limit=300,  # 5 minute time limit for debugging
    )


@task
def exercism_python(harness: str = "opencode") -> Task:
    """
    Exercism: Python coding tasks only.

    Returns:
        Task configured for Python-only Exercism evaluation
    """
    return exercism(languages=["python"], harness=harness)


@task
def exercism_javascript(harness: str = "opencode") -> Task:
    """
    Exercism: JavaScript coding tasks only.

    Returns:
        Task configured for JavaScript-only Exercism evaluation
    """
    return exercism(languages=["javascript"], harness=harness)


@task
def exercism_go(harness: str = "opencode") -> Task:
    """
    Exercism: Go coding tasks only.

    Returns:
        Task configured for Go-only Exercism evaluation
    """
    return exercism(languages=["go"], harness=harness)


@task
def exercism_java(harness: str = "opencode") -> Task:
    """
    Exercism: Java coding tasks only.

    Returns:
        Task configured for Java-only Exercism evaluation
    """
    return exercism(languages=["java"], harness=harness)


@task
def exercism_rust(harness: str = "opencode") -> Task:
    """
    Exercism: Rust coding tasks only.

    Returns:
        Task configured for Rust-only Exercism evaluation
    """
    return exercism(languages=["rust"], harness=harness)
