"""
Exercism evaluation tasks.

This module provides inspect_ai Tasks for evaluating coding abilities across
multiple programming languages using the Exercism benchmark.
"""

from typing import Optional, List
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from openbench.datasets.exercism import get_exercism_dataset
from openbench.solvers.exercism_solver import exercism_solver
from openbench.scorers.exercism import exercism_scorer
from openbench.agents import AgentManager
from openbench.agents.docker_manager import DockerManager


TASK_DIR = Path(__file__).parent
COMPOSE_PATH = (TASK_DIR / "compose.yaml").resolve()


@task
def exercism(
    languages: Optional[List[str]] = None,
    code_agent: str = "opencode",
) -> Task:
    """
    Exercism: Multi-language coding benchmark.

    Evaluates coding abilities across multiple programming languages using
    real-world coding exercises from the Exercism Tasks.

    Args:
        languages: List of programming languages to include (python, go, javascript, java, rust).
                  If None, includes all supported languages.
        code_agent: CLI code agent to use for code evaluation.
                   Defaults to 'opencode'. Can also be set via --code-agent flag.
                   Valid options: aider, opencode, claude, roo

    Returns:
        Task configured for Exercism evaluation
    """
    # Validate code agent
    if not AgentManager.validate_code_agent(code_agent):
        valid_agents = AgentManager.get_valid_code_agents()
        raise ValueError(
            f"Invalid code agent: {code_agent}. Valid options: {', '.join(valid_agents)}"
        )

    # Generate dynamic Docker files for this specific agent
    DockerManager.generate_eval_files(code_agent, TASK_DIR)

    dataset = get_exercism_dataset(languages=languages)

    # Add code agent to each sample's metadata so the solver can access it
    for sample in dataset:
        if not hasattr(sample, "metadata") or sample.metadata is None:
            sample.metadata = {}
        sample.metadata["code_agent"] = code_agent

    # Determine task name based on languages
    if languages and len(languages) == 1:
        task_name = f"exercism_{languages[0]}"
    else:
        task_name = "exercism"

    return Task(
        name=task_name,
        dataset=dataset,
        solver=exercism_solver(),
        scorer=exercism_scorer(),
        sandbox=("docker", str(COMPOSE_PATH)),
        config=GenerateConfig(
            max_tokens=4096,
        ),
        time_limit=360,  # 6 minute time limit
    )


@task
def exercism_python(code_agent: str = "opencode") -> Task:
    """
    Exercism: Python coding tasks only.

    Returns:
        Task configured for Python-only Exercism evaluation
    """
    return exercism(languages=["python"], code_agent=code_agent)


@task
def exercism_javascript(code_agent: str = "opencode") -> Task:
    """
    Exercism: JavaScript coding tasks only.

    Returns:
        Task configured for JavaScript-only Exercism evaluation
    """
    return exercism(languages=["javascript"], code_agent=code_agent)


@task
def exercism_go(code_agent: str = "opencode") -> Task:
    """
    Exercism: Go coding tasks only.

    Returns:
        Task configured for Go-only Exercism evaluation
    """
    return exercism(languages=["go"], code_agent=code_agent)


@task
def exercism_java(code_agent: str = "opencode") -> Task:
    """
    Exercism: Java coding tasks only.

    Returns:
        Task configured for Java-only Exercism evaluation
    """
    return exercism(languages=["java"], code_agent=code_agent)


@task
def exercism_rust(code_agent: str = "opencode") -> Task:
    """
    Exercism: Rust coding tasks only.

    Returns:
        Task configured for Rust-only Exercism evaluation
    """
    return exercism(languages=["rust"], code_agent=code_agent)
