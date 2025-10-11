from openbench.datasets.smt import (
    get_smt_dataset,
    get_smt_algebra_dataset,
    get_smt_calculus_dataset,
    get_smt_discrete_dataset,
    get_smt_general_dataset,
    get_smt_geometry_dataset,
    get_smt_guts_dataset,
)
from openbench.scorers.smt import smt_scorer
from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig


@task
def smt() -> Task:
    """
    SMT (Stanford Math Tournament) 2024 evaluation task.

    This benchmark evaluates language models on problems from the Stanford Math
    Tournament 2024, covering Algebra, Calculus, Discrete Math, General Math,
    Geometry, and Guts rounds.

    Returns:
        Task object configured for SMT evaluation
    """
    return Task(
        dataset=get_smt_dataset(),
        solver=[generate()],
        scorer=smt_scorer(),
        config=GenerateConfig(
            max_tokens=8192,
            temperature=0.0,
        ),
    )


@task
def smt_algebra() -> Task:
    """
    SMT Algebra evaluation task.

    This benchmark evaluates language models on Algebra problems from the
    Stanford Math Tournament 2024.

    Returns:
        Task object configured for SMT Algebra evaluation
    """
    return Task(
        dataset=get_smt_algebra_dataset(),
        solver=[generate()],
        scorer=smt_scorer(),
        config=GenerateConfig(
            max_tokens=8192,
            temperature=0.0,
        ),
    )


@task
def smt_calculus() -> Task:
    """
    SMT Calculus evaluation task.

    This benchmark evaluates language models on Calculus problems from the
    Stanford Math Tournament 2024.

    Returns:
        Task object configured for SMT Calculus evaluation
    """
    return Task(
        dataset=get_smt_calculus_dataset(),
        solver=[generate()],
        scorer=smt_scorer(),
        config=GenerateConfig(
            max_tokens=8192,
            temperature=0.0,
        ),
    )


@task
def smt_discrete() -> Task:
    """
    SMT Discrete Math evaluation task.

    This benchmark evaluates language models on Discrete Math problems from the
    Stanford Math Tournament 2024.

    Returns:
        Task object configured for SMT Discrete evaluation
    """
    return Task(
        dataset=get_smt_discrete_dataset(),
        solver=[generate()],
        scorer=smt_scorer(),
        config=GenerateConfig(
            max_tokens=8192,
            temperature=0.0,
        ),
    )


@task
def smt_general() -> Task:
    """
    SMT General Math evaluation task.

    This benchmark evaluates language models on General Math problems from the
    Stanford Math Tournament 2024.

    Returns:
        Task object configured for SMT General evaluation
    """
    return Task(
        dataset=get_smt_general_dataset(),
        solver=[generate()],
        scorer=smt_scorer(),
        config=GenerateConfig(
            max_tokens=8192,
            temperature=0.0,
        ),
    )


@task
def smt_geometry() -> Task:
    """
    SMT Geometry evaluation task.

    This benchmark evaluates language models on Geometry problems from the
    Stanford Math Tournament 2024.

    Returns:
        Task object configured for SMT Geometry evaluation
    """
    return Task(
        dataset=get_smt_geometry_dataset(),
        solver=[generate()],
        scorer=smt_scorer(),
        config=GenerateConfig(
            max_tokens=8192,
            temperature=0.0,
        ),
    )


@task
def smt_guts() -> Task:
    """
    SMT Guts Round evaluation task.

    This benchmark evaluates language models on Guts round problems from the
    Stanford Math Tournament 2024.

    Returns:
        Task object configured for SMT Guts evaluation
    """
    return Task(
        dataset=get_smt_guts_dataset(),
        solver=[generate()],
        scorer=smt_scorer(),
        config=GenerateConfig(
            max_tokens=8192,
            temperature=0.0,
        ),
    )
